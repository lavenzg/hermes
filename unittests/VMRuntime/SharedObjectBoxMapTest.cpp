/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/VM/BoxedSharedObject.h"

#include "VMRuntimeTestHelpers.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

using namespace hermes::vm;

namespace {

using SharedObjectBoxMapTest = RuntimeTestFixture;

/// Fabricated addresses. The map only ever compares and hashes its keys, so a
/// real shared allocation would add a dependency without adding coverage.
static SharedGCCell *fakeCell(uintptr_t n) {
  return reinterpret_cast<SharedGCCell *>(n);
}

TEST_F(SharedObjectBoxMapTest, Find_OnAnUnknownCellIsNull) {
  EXPECT_EQ(nullptr, runtime.findBoxForSharedCell(fakeCell(0x1000)));
}

TEST_F(SharedObjectBoxMapTest, Find_ReturnsTheBoxThatWasRegistered) {
  // The deduplication invariant: one cell, one box, so `===` on a shared object
  // is true of itself.
  GCScope scope{runtime, "SharedObjectBoxMapTest"};
  auto *cell = fakeCell(0x1000);
  auto box = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
  runtime.setBoxForSharedCell(cell, box);

  EXPECT_EQ(box.get(), runtime.findBoxForSharedCell(cell));
  // And a different cell is still unknown.
  EXPECT_EQ(nullptr, runtime.findBoxForSharedCell(fakeCell(0x2000)));
}

TEST_F(SharedObjectBoxMapTest, Entries_AreVisitedOnEveryWeakRootPass) {
  // The map's WeakRoots are known to the collector through one loop in
  // markWeakRoots and nowhere else, so an entry it skips keeps a pointer the
  // collector never had the chance to update. A young-gen collection moves the
  // box, so the map must come back pointing at where it moved to.
  GCScope scope{runtime, "SharedObjectBoxMapTest"};
  auto *cell = fakeCell(0x3000);
  auto box = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
  runtime.setBoxForSharedCell(cell, box);
  ASSERT_EQ(box.get(), runtime.findBoxForSharedCell(cell));

  runtime.collect("test");

  // The handle is updated by the collector; the map has to agree with it.
  EXPECT_EQ(box.get(), runtime.findBoxForSharedCell(cell));
  EXPECT_EQ(cell, box->getSharedCell());
}

TEST_F(SharedObjectBoxMapTest, Find_IsNullOnceTheBoxIsCollected) {
  auto *cell = fakeCell(0x4000);
  {
    GCScope scope{runtime, "SharedObjectBoxMapTest"};
    auto box = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
    runtime.setBoxForSharedCell(cell, box);
    ASSERT_NE(nullptr, runtime.findBoxForSharedCell(cell));
  }
  // Nothing holds the box now.
  runtime.collect("test");

  EXPECT_EQ(nullptr, runtime.findBoxForSharedCell(cell))
      << "the map kept a dead box alive, or handed one out";
}

TEST_F(SharedObjectBoxMapTest, DeadEntries_ArePrunedOnTheNextAccess) {
  // Mark now, erase later: the weak-root pass only nulls, and the sweep happens
  // here. Erasing inside the pass is not an option -- it also runs from
  // checkWellFormed with a read-only acceptor, on by default in Debug.
  auto *dead = fakeCell(0x5000);
  auto *live = fakeCell(0x6000);

  GCScope outer{runtime, "SharedObjectBoxMapTest"};
  auto keep = runtime.makeHandle(BoxedSharedObject::create(runtime, live));
  runtime.setBoxForSharedCell(live, keep);
  {
    GCScope scope{runtime, "SharedObjectBoxMapTest"};
    auto doomed = runtime.makeHandle(BoxedSharedObject::create(runtime, dead));
    runtime.setBoxForSharedCell(dead, doomed);
  }
  ASSERT_EQ(2u, runtime.sharedObjectBoxCountSlow());

  runtime.collect("test");
  // The pass nulls but does not erase, so the entry is still there.
  ASSERT_EQ(2u, runtime.sharedObjectBoxCountSlow());

  // Any access sweeps it, and leaves the live one alone.
  EXPECT_EQ(nullptr, runtime.findBoxForSharedCell(dead));
  EXPECT_EQ(1u, runtime.sharedObjectBoxCountSlow());
  EXPECT_EQ(keep.get(), runtime.findBoxForSharedCell(live));
}

TEST_F(SharedObjectBoxMapTest, Insert_DoesNotPrune) {
  // Pruning is a lookup-only side effect. An insert runs under the shared
  // heap's allocation lock, and while marking is on the prune pushes each key
  // it erases into the barrier ring, which can reach the spill path and so the
  // barrier mutex -- the reverse of the lock order, which puts the barrier
  // mutex first. The debug assert on the way in must not reach it either,
  // which is the only way this could creep back.
  auto *dead = fakeCell(0x8000);
  auto *fresh = fakeCell(0x9000);

  GCScope outer{runtime, "SharedObjectBoxMapTest"};
  {
    GCScope scope{runtime, "SharedObjectBoxMapTest"};
    auto doomed = runtime.makeHandle(BoxedSharedObject::create(runtime, dead));
    runtime.setBoxForSharedCell(dead, doomed);
  }
  runtime.collect("test");
  ASSERT_EQ(1u, runtime.sharedObjectBoxCountSlow());

  auto keep = runtime.makeHandle(BoxedSharedObject::create(runtime, fresh));
  runtime.setBoxForSharedCell(fresh, keep);

  EXPECT_EQ(2u, runtime.sharedObjectBoxCountSlow())
      << "the insert pruned; that side effect belongs to lookup alone";
}

TEST_F(SharedObjectBoxMapTest, Rebinding_AfterACollectionIsAllowed) {
  // A cell whose box died can be boxed again -- the cell is still perfectly
  // alive, this runtime just stopped naming it for a while.
  auto *cell = fakeCell(0x7000);
  {
    GCScope scope{runtime, "SharedObjectBoxMapTest"};
    auto first = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
    runtime.setBoxForSharedCell(cell, first);
  }
  runtime.collect("test");
  ASSERT_EQ(nullptr, runtime.findBoxForSharedCell(cell));

  GCScope scope{runtime, "SharedObjectBoxMapTest"};
  auto second = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
  runtime.setBoxForSharedCell(cell, second);
  EXPECT_EQ(second.get(), runtime.findBoxForSharedCell(cell));
}

TEST_F(SharedObjectBoxMapTest, RootKeys_AreEveryEntryIncludingDeadOnes) {
  // This is what the shared collector marks from. A key whose box has died but
  // has not been pruned must still be included: over-retaining a cell for one
  // cycle is safe, dropping one is a use-after-free.
  auto *live = fakeCell(0xA000);
  auto *dead = fakeCell(0xB000);

  GCScope outer{runtime, "SharedObjectBoxMapTest"};
  auto keep = runtime.makeHandle(BoxedSharedObject::create(runtime, live));
  runtime.setBoxForSharedCell(live, keep);
  {
    GCScope scope{runtime, "SharedObjectBoxMapTest"};
    auto doomed = runtime.makeHandle(BoxedSharedObject::create(runtime, dead));
    runtime.setBoxForSharedCell(dead, doomed);
  }
  // Nulls the dead entry's WeakRoot without erasing it.
  runtime.collect("test");
  ASSERT_EQ(2u, runtime.sharedObjectBoxCountSlow());

  // Pre-seeded, because it appends rather than assigns -- the collector
  // accumulates across runtimes.
  std::vector<SharedGCCell *> keys{fakeCell(0xC000)};
  runtime.appendSharedRootKeys(keys);

  ASSERT_EQ(3u, keys.size());
  EXPECT_EQ(fakeCell(0xC000), keys[0]) << "it replaced instead of appending";
  EXPECT_NE(keys.end(), std::find(keys.begin(), keys.end(), live));
  EXPECT_NE(keys.end(), std::find(keys.begin(), keys.end(), dead))
      << "a dead but unpruned key was dropped from the root set";
}

TEST_F(SharedObjectBoxMapTest, RootKeys_ReadWhileTheOwnerMutatesTheMap) {
  // The lock's entire purpose, and it proves nothing single-threaded -- it is
  // here for TSan and for ASan's use-after-free. The reader stands in for the
  // collector: another thread walking the keys while this one inserts, which
  // rehashes and frees the old bucket array, and prunes, which erases.
  //
  // It also exercises the claim that the weak-root walk needs no lock: the
  // collections below forward every entry's value while the reader is reading
  // keys.
  std::atomic<bool> stop{false};
  std::thread reader([&] {
    std::vector<SharedGCCell *> keys;
    while (!stop.load(std::memory_order_acquire)) {
      keys.clear();
      runtime.appendSharedRootKeys(keys);
    }
  });

  {
    GCScope scope{runtime, "SharedObjectBoxMapTest"};
    for (uintptr_t i = 1; i <= 512; ++i) {
      GCScopeMarkerRAII marker{runtime};
      auto *cell = fakeCell(i * 0x100);
      auto box = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
      runtime.setBoxForSharedCell(cell, box);
      // Every 64 boxes, a collection finds the earlier ones dead and forwards
      // the live one's value while the reader reads keys. The lookup after it
      // prunes, which is the erase half of what the lock guards.
      if (i % 64 == 0)
        runtime.collect("test");
      runtime.findBoxForSharedCell(cell);
    }
  }
  stop.store(true, std::memory_order_release);
  reader.join();

  EXPECT_LT(runtime.sharedObjectBoxCountSlow(), 512u) << "nothing was pruned";
}

} // namespace
