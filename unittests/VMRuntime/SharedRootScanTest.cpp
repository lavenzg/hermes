/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/BoxedSharedObject.h"
#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedMutator.h"

#include "VMRuntimeTestHelpers.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

namespace {

using namespace hermes;
using namespace hermes::vm;

constexpr size_t kAmpleBudget = 8 << 20;

/// Free everything. Two sweeps always suffice: cells are born marked, so the
/// first clears the mark on whatever survives and the second collects it.
static void freeAll(SharedGC &gc) {
  gc.sweep();
  gc.sweep();
}

/// Two runtimes, because one proves nothing: the scan's job is to union every
/// attached runtime's map, and a single-runtime fixture cannot tell that apart
/// from reading one map twice. The existing shared-heap fixtures all name one
/// runtime, which is why this test has its own.
struct SharedRootScanTest : public ::testing::Test {
  SharedGC gc{kAmpleBudget};
  std::shared_ptr<Runtime> rtA{Runtime::create(kTestRTConfig)};
  std::shared_ptr<Runtime> rtB{Runtime::create(kTestRTConfig)};
  Runtime &a{*rtA};
  Runtime &b{*rtB};

  ~SharedRootScanTest() override {
    freeAll(gc);
  }

  SharedArray *array() {
    SharedArray *arr = SharedArray::create(gc, 1);
    EXPECT_NE(nullptr, arr);
    return arr;
  }

  /// Give \p runtime a box for \p cell, as the materialiser will.
  void box(Runtime &runtime, SharedGCCell *cell) {
    GCScope scope{runtime, "SharedRootScanTest"};
    auto handle = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
    runtime.setBoxForSharedCell(cell, handle);
  }
};

TEST_F(SharedRootScanTest, Scan_MarksEveryAttachedRuntimesKeys) {
  SharedMutator mutatorA{gc, a};
  SharedMutator mutatorB{gc, b};
  auto *onlyA = array();
  auto *onlyB = array();
  box(a, onlyA);
  box(b, onlyB);

  gc.beginCycle();
  EXPECT_EQ(2u, gc.markRuntimeRoots());
  gc.drainMarkWorklist();

  EXPECT_TRUE(SharedGC::isMarked(onlyA));
  EXPECT_TRUE(SharedGC::isMarked(onlyB))
      << "the scan stopped at the first runtime";
  gc.endCycle();
}

TEST_F(SharedRootScanTest, Scan_ReportsAKeyWhoseBoxHasDied) {
  // Over-retention, on purpose. The box is gone, so the runtime can no longer
  // name the cell, but the key survives until the prune -- and scanning it is
  // what keeps a cycle from freeing a cell another runtime is mid-acquisition
  // on. Dropping it would be the unsafe direction.
  SharedMutator mutatorA{gc, a};
  auto *cell = array();
  box(a, cell);
  // Nulls the WeakRoot without erasing the entry.
  a.collect("test");
  // Deliberately not findBoxForSharedCell to check the box died: that lookup
  // prunes, which would erase the very key under test. The entry surviving the
  // pass is what "nulled but not pruned" looks like from outside.
  ASSERT_EQ(1u, a.sharedObjectBoxCountSlow());

  gc.beginCycle();
  EXPECT_EQ(1u, gc.markRuntimeRoots());
  gc.drainMarkWorklist();

  EXPECT_TRUE(SharedGC::isMarked(cell))
      << "a key whose box died was dropped from the root set";
  gc.endCycle();

  // Only now, once the scan is done: this both proves the box really was dead
  // and prunes the entry, which is the mutator-side half of the same story.
  EXPECT_EQ(nullptr, a.findBoxForSharedCell(cell));
  EXPECT_EQ(0u, a.sharedObjectBoxCountSlow());
}

TEST_F(SharedRootScanTest, Scan_IsIdempotentAndPicksUpLaterBoxes) {
  // Re-scanning has to be free, because the intended driver folds it into the
  // certification loop rather than running it once.
  SharedMutator mutatorA{gc, a};
  auto *first = array();
  box(a, first);

  gc.beginCycle();
  ASSERT_EQ(1u, gc.markRuntimeRoots());
  gc.drainMarkWorklist();
  ASSERT_TRUE(SharedGC::isMarked(first));

  // Black allocation covers this one, but it also joins the map, so the second
  // scan must see it and must not choke on the already-marked first.
  auto *second = array();
  box(a, second);
  EXPECT_EQ(2u, gc.markRuntimeRoots());
  gc.drainMarkWorklist();

  EXPECT_TRUE(SharedGC::isMarked(first));
  EXPECT_TRUE(SharedGC::isMarked(second));
  gc.endCycle();
}

TEST_F(SharedRootScanTest, Scan_CopiesUnderTheMapLockAlone) {
  // While the scan copies a map it holds that map's lock, which keeps the map
  // alive and still, and not barrierMtx_, which every spill takes.
  SharedMutator mutatorA{gc, a};
  SharedMutator mutatorB{gc, b};
  auto *cell = array();
  box(b, cell);

  std::atomic<bool> copying{false};
  std::atomic<bool> spilled{false};
  bool sawCopy = false;
  bool mapLockHeld = false;
  bool spilledDuringCopy = false;
  gc.beginCycle();
  std::thread spiller([&] {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!copying.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    sawCopy = copying.load(std::memory_order_acquire);
    if (sawCopy) {
      std::mutex &mapLock = b.sharedObjectBoxesLock();
      mapLockHeld = !mapLock.try_lock();
      if (!mapLockHeld)
        mapLock.unlock();
    }
    for (uint32_t i = 0; i <= SharedBarrierRing::kCapacity; ++i)
      mutatorA.push(cell, gc.cycleSeq());
    spilled.store(true, std::memory_order_release);
  });
  const auto beforeCopy = [&](const Runtime &runtime) {
    if (&runtime != &b)
      return;
    copying.store(true, std::memory_order_release);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!spilled.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    spilledDuringCopy = spilled.load(std::memory_order_acquire);
  };
  EXPECT_EQ(1u, gc.markRuntimeRoots(beforeCopy));
  spiller.join();

  EXPECT_TRUE(sawCopy) << "the scan never copied b's map";
  EXPECT_TRUE(mapLockHeld) << "the scan copied a map without holding its lock";
  EXPECT_TRUE(spilledDuringCopy) << "a spill waited for the scan's copy";
  gc.drainBarrierWorklist();
  gc.drainMarkWorklist();
  gc.endCycle();
}

TEST_F(SharedRootScanTest, Scan_LetsARuntimeSpillWhileAMapIsBusy) {
  // Every spill takes barrierMtx_, so the scan holds it only to find a runtime
  // and try its map lock. Waiting there for a busy map would stall every
  // runtime's spills behind whatever holds that map; skipping the map would
  // drop its keys.
  SharedMutator mutatorA{gc, a};
  SharedMutator mutatorB{gc, b};
  auto *cell = array();
  box(b, cell);

  std::atomic<bool> busy{false};
  std::atomic<bool> spilled{false};
  bool spilledWhileBusy = false;
  // Holds b's map lock until a spill gets through, or for ten seconds.
  std::thread holder([&] {
    std::lock_guard<std::mutex> lk(b.sharedObjectBoxesLock());
    busy.store(true, std::memory_order_release);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!spilled.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    spilledWhileBusy = spilled.load(std::memory_order_acquire);
  });
  while (!busy.load(std::memory_order_acquire))
    std::this_thread::yield();

  gc.beginCycle();
  std::thread spiller([&] {
    // Long enough for the scan to reach b's map first.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    for (uint32_t i = 0; i <= SharedBarrierRing::kCapacity; ++i)
      mutatorA.push(cell, gc.cycleSeq());
    spilled.store(true, std::memory_order_release);
  });
  EXPECT_EQ(1u, gc.markRuntimeRoots()) << "the scan skipped a busy map";
  spiller.join();
  holder.join();

  EXPECT_TRUE(spilledWhileBusy)
      << "a spill waited for a scan that was waiting for a busy map";
  gc.drainBarrierWorklist();
  gc.drainMarkWorklist();
  gc.endCycle();
}

TEST_F(SharedRootScanTest, Scan_SkipsARuntimeThatLeavesWhileItsMapIsBusy) {
  // The scan checks again that a runtime is registered each time it retries a
  // busy map. One that has left is skipped: its map may be gone.
  SharedMutator mutatorA{gc, a};
  auto mutatorB = std::make_unique<SharedMutator>(gc, b);
  auto *cell = array();
  box(b, cell);

  std::atomic<bool> busy{false};
  std::atomic<bool> scanned{false};
  bool scannedWhileBusy = false;
  // Holds b's map lock until the scan returns, or for ten seconds.
  std::thread holder([&] {
    std::lock_guard<std::mutex> lk(b.sharedObjectBoxesLock());
    busy.store(true, std::memory_order_release);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!scanned.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    scannedWhileBusy = scanned.load(std::memory_order_acquire);
  });
  while (!busy.load(std::memory_order_acquire))
    std::this_thread::yield();

  gc.beginCycle();
  std::thread leaver([&] {
    // Long enough for the scan to be retrying b's map.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    mutatorB.reset();
  });
  EXPECT_EQ(0u, gc.markRuntimeRoots())
      << "the scan read a runtime that had left";
  scanned.store(true, std::memory_order_release);
  // The holder first: the leaver waits for b's map lock once it is unlinked.
  holder.join();
  leaver.join();

  EXPECT_TRUE(scannedWhileBusy)
      << "the scan kept trying the map of a runtime that had left";
  gc.drainBarrierWorklist();
  gc.drainMarkWorklist();
  gc.endCycle();
}

TEST_F(SharedRootScanTest, Scan_RacesAttachAndDetach) {
  // What barrierMtx_ is for here. Registration pushes onto mutators_ and
  // deregistration swaps-and-pops it, so walking that vector unguarded while
  // runtimes come and go is a use-after-free or a skipped entry. Proves
  // nothing single-threaded -- it is here for TSan.
  auto *cell = array();
  box(a, cell);
  SharedMutator resident{gc, a};

  std::atomic<bool> stop{false};
  std::thread churn([&] {
    while (!stop.load(std::memory_order_acquire)) {
      SharedMutator transient{gc, b};
      (void)transient;
    }
  });

  gc.beginCycle();
  for (int i = 0; i < 500; ++i)
    gc.markRuntimeRoots();
  gc.drainMarkWorklist();
  stop.store(true, std::memory_order_release);
  churn.join();

  // The resident runtime's key is found regardless of the churn around it.
  EXPECT_TRUE(SharedGC::isMarked(cell));
  gc.endCycle();
}

TEST_F(SharedRootScanTest, Detach_WaitsForACopyOfItsKeysToFinish) {
  // The scan copies a map under that map's lock alone, so what keeps a
  // detaching runtime's map alive through a copy is unregisterMutator taking
  // the lock after the unlink. A thread holding the lock stands in for a scan
  // copying. Marking is on, so a leaving runtime's keys go too.
  auto *cell = array();
  box(b, cell);
  auto mutatorB = std::make_unique<SharedMutator>(gc, b);
  gc.beginCycle();

  std::atomic<bool> busy{false};
  std::atomic<bool> release{false};
  std::atomic<bool> left{false};
  bool releasedOnTime = false;
  // Holds b's map lock until told to let go, or for twenty seconds. It takes
  // no other lock, so the poll below never waits on it.
  std::thread holder([&] {
    std::lock_guard<std::mutex> lk(b.sharedObjectBoxesLock());
    busy.store(true, std::memory_order_release);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!release.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    releasedOnTime = release.load(std::memory_order_acquire);
  });
  while (!busy.load(std::memory_order_acquire))
    std::this_thread::yield();
  std::thread leaver([&] {
    mutatorB.reset();
    left.store(true, std::memory_order_release);
  });

  // Unlinked, so no scan can start a copy from here on.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (gc.mutatorCountSlow() != 0 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  const bool unlinked = gc.mutatorCountSlow() == 0;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const bool leftWhileCopying = left.load(std::memory_order_acquire);
  release.store(true, std::memory_order_release);
  holder.join();
  leaver.join();

  EXPECT_TRUE(releasedOnTime) << "the poll waited for the map lock";
  EXPECT_TRUE(unlinked) << "the runtime waited for the copy before unlinking";
  EXPECT_FALSE(leftWhileCopying)
      << "a runtime finished leaving while its keys were being copied";
  EXPECT_TRUE(left.load(std::memory_order_acquire));
  gc.drainBarrierWorklist();
  gc.drainMarkWorklist();
  gc.endCycle();
}

} // namespace
