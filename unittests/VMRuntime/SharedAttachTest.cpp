/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/BoxedSharedObject.h"
#include "hermes/VM/JSArray.h"
#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedGCBoxing-inline.h"
#include "hermes/VM/SharedHeap/SharedMutator.h"

#include "VMRuntimeTestHelpers.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace {

using namespace hermes;
using namespace hermes::vm;

constexpr size_t kAmpleBudget = 8 << 20;

/// A shared heap plus runtimes attached to it through Runtime::create. A test
/// that must hold a barrier window open adds one mutator built by hand, for a
/// runtime created without the heap, since an attached runtime's own mutator
/// is private.
struct SharedAttachTest : public ::testing::Test {
  std::shared_ptr<SharedGC> gc{std::make_shared<SharedGC>(kAmpleBudget)};

  std::shared_ptr<Runtime> attach() {
    return Runtime::create(kTestRTConfig, gc);
  }

  ~SharedAttachTest() override {
    // Runtimes are test locals and already gone, so the registry is empty and
    // a cycle with no roots frees the heap.
    gc->collect();
  }

  /// \return a new array of \p length empty slots, boxed by no runtime.
  SharedArray *array(uint32_t length) {
    SharedArray *arr = SharedArray::create(*gc, length);
    EXPECT_NE(nullptr, arr);
    return arr;
  }

  /// Allocate a one-slot array and its box in \p runtime, rooting the box in
  /// \p rooted.
  /// \return the array, or null if the allocation failed.
  SharedArray *boxedArray(
      Runtime &runtime,
      PinnedValue<BoxedSharedObject> &rooted) {
    rooted = gc->makeBoxed<SharedArray>(
        runtime, SharedArray::allocationSize(1), uint32_t{1});
    return *rooted ? vmcast<SharedArray>(rooted->getSharedCell()) : nullptr;
  }

  /// Give \p runtime a box for \p cell, as the materialiser will. Nothing
  /// roots the box, but the key names the cell until a prune drops it.
  static void box(Runtime &runtime, SharedGCCell *cell) {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> handle;
    } lv;
    LocalsRAII lraii{runtime, &lv};
    lv.handle = BoxedSharedObject::create(runtime, cell);
    runtime.setBoxForSharedCell(cell, lv.handle);
  }

  /// What collect() does once its first wait is over, up to the round that
  /// certifies.
  void traceToCertification() {
    EXPECT_TRUE(gc->pollInFlightBarriers());
    gc->beginSecondInFlightPass();
    EXPECT_TRUE(gc->pollInFlightBarriers()) << "no window is open by now";
    gc->markRuntimeRoots();
    do {
      gc->drainMarkWorklist();
    } while (!gc->certifyRound());
  }
};

TEST_F(SharedAttachTest, Attach_RegistersAndDeregisters) {
  EXPECT_EQ(0u, gc->mutatorCountSlow());
  {
    auto rt = attach();
    EXPECT_EQ(1u, gc->mutatorCountSlow());
    {
      auto rt2 = attach();
      EXPECT_EQ(2u, gc->mutatorCountSlow());
    }
    EXPECT_EQ(1u, gc->mutatorCountSlow()) << "the second did not deregister";
  }
  EXPECT_EQ(0u, gc->mutatorCountSlow());
}

TEST_F(SharedAttachTest, Attach_IsOptional) {
  // A runtime created with no shared heap must be unaffected by one existing.
  auto plain = Runtime::create(kTestRTConfig);
  EXPECT_EQ(0u, gc->mutatorCountSlow());
}

TEST_F(SharedAttachTest, Deregistration_BeatsMemberTeardown) {
  // The drop belongs in a post-shutdown deleter, not left to the member's own
  // destructor. Both deregister eventually, so only the timing tells them
  // apart: deleters run in registration order, so one registered here runs
  // after the runtime's own and can see that it already happened.
  size_t countInsideDeleter = 99;
  {
    auto rt = attach();
    rt->addPostShutdownDeleter(
        [&] { countInsideDeleter = gc->mutatorCountSlow(); });
    ASSERT_EQ(1u, gc->mutatorCountSlow());
  }
  EXPECT_EQ(0u, countInsideDeleter)
      << "deregistration did not happen in the runtime's own post-shutdown "
         "deleter";
}

TEST_F(SharedAttachTest, Collect_EndToEndThroughRealRuntimes) {
  // The whole thing, with nothing hand-wired: two attached runtimes and cells
  // allocated and boxed through makeBoxed. collect() needs nothing from either
  // runtime: neither is inside a barrier, so it does not wait.
  auto rtA = attach();
  auto rtB = attach();
  Runtime &a = *rtA;
  Runtime &b = *rtB;

  SharedArray *keptByA = nullptr;
  SharedArray *keptByB = nullptr;
  {
    GCScope scope{a, "SharedAttachTest"};
    auto box = a.makeHandle(gc->makeBoxed<SharedArray>(
        a, SharedArray::allocationSize(1), uint32_t{1}));
    ASSERT_TRUE(*box);
    keptByA = vmcast<SharedArray>(box->getSharedCell());
  }
  {
    GCScope scope{b, "SharedAttachTest"};
    auto box = b.makeHandle(gc->makeBoxed<SharedArray>(
        b, SharedArray::allocationSize(1), uint32_t{1}));
    ASSERT_TRUE(*box);
    keptByB = vmcast<SharedArray>(box->getSharedCell());
  }
  // Named by nobody.
  ASSERT_NE(nullptr, SharedArray::create(*gc, 1));
  ASSERT_EQ(3u, gc->liveCellCountSlow());

  gc->collect();

  EXPECT_EQ(2u, gc->liveCellCountSlow())
      << "the unnamed cell should have been the only casualty";
  EXPECT_EQ(1u, keptByA->length());
  EXPECT_EQ(1u, keptByB->length());
}

TEST_F(SharedAttachTest, Detach_DuringTheWait_RetainsACellStoredInABlackCell) {
  // A boxes `cell` before the cycle. While collect() would be waiting on B's
  // window, A stores `cell` into `holder`, which is born marked and so never
  // traced, B boxes `holder`, and A detaches. A's keys are the only record of
  // `cell`.
  auto rtA = attach();
  auto rtB = Runtime::create(kTestRTConfig);
  // By hand, because an attached runtime's own mutator is private and B's
  // window has to stay open.
  SharedMutator mutatorB{*gc, *rtB};

  SharedArray *cell = nullptr;
  SharedArray *holder = nullptr;
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{*rtA, &lv};
    cell = boxedArray(*rtA, lv.box);
    ASSERT_NE(nullptr, cell);

    mutatorB.enterBarrier();
    gc->beginCycle();
    // EXPECT, not ASSERT, until the cycle ends: returning early would leave
    // B's window and the cycle open.
    EXPECT_FALSE(gc->pollInFlightBarriers()) << "B's window holds the wait";
    EXPECT_FALSE(SharedGC::isMarked(cell)) << "the reset whitened it";
    holder = array(1);
    EXPECT_TRUE(SharedGC::isMarked(holder)) << "born marked, never traced";
    // initField: nothing else can reach `holder` yet.
    holder->initField(0, SharedValue::encodePointer(cell));
    box(*rtB, holder);
  }
  // The box is dead and its entry cleared, but the key stays until a prune,
  // and leaving must hand it over all the same.
  rtA->collect("test");
  EXPECT_EQ(1u, rtA->sharedObjectBoxCountSlow());
  rtA.reset();
  EXPECT_EQ(1u, gc->mutatorCountSlow()) << "A did not detach";
  mutatorB.exitBarrier();

  traceToCertification();
  EXPECT_TRUE(SharedGC::isMarked(cell))
      << "a runtime that left during the wait took its keys with it";
  gc->endCycle();
  gc->sweep();

  // Counted before `cell` is read: a swept cell's slot is poisoned.
  ASSERT_EQ(2u, gc->liveCellCountSlow()) << "cell was swept";
  EXPECT_EQ(cell, holder->at(0).getPointer());
  EXPECT_EQ(1u, cell->length());
}

TEST_F(SharedAttachTest, Prune_DuringTheWait_RetainsACellStoredInABlackCell) {
  // As above, but A stays attached and erases the key instead: once `cell` is
  // in `holder`, A's box for it dies, a local GC clears the entry, and a
  // lookup prunes the key, all before the root scan reaches A.
  auto rtA = attach();
  auto rtB = Runtime::create(kTestRTConfig);
  Runtime &a = *rtA;
  SharedMutator mutatorB{*gc, *rtB};

  SharedArray *cell = nullptr;
  SharedArray *holder = nullptr;
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{a, &lv};
    cell = boxedArray(a, lv.box);
    ASSERT_NE(nullptr, cell);

    mutatorB.enterBarrier();
    gc->beginCycle();
    EXPECT_FALSE(gc->pollInFlightBarriers()) << "B's window holds the wait";
    EXPECT_FALSE(SharedGC::isMarked(cell)) << "the reset whitened it";
    holder = array(1);
    EXPECT_TRUE(SharedGC::isMarked(holder)) << "born marked, never traced";
    // While A's box for `cell` is still alive.
    holder->initField(0, SharedValue::encodePointer(cell));
    box(*rtB, holder);
  }
  a.collect("test");
  EXPECT_EQ(nullptr, a.findBoxForSharedCell(cell)) << "the box survived";
  EXPECT_EQ(0u, a.sharedObjectBoxCountSlow()) << "the lookup did not prune";
  EXPECT_EQ(2u, gc->mutatorCountSlow()) << "A must still be attached";
  mutatorB.exitBarrier();

  traceToCertification();
  EXPECT_TRUE(SharedGC::isMarked(cell))
      << "erasing a key during the wait left no record of its cell";
  gc->endCycle();
  gc->sweep();

  ASSERT_EQ(2u, gc->liveCellCountSlow()) << "cell was swept";
  EXPECT_EQ(cell, holder->at(0).getPointer());
  EXPECT_EQ(1u, cell->length());
}

TEST_F(SharedAttachTest, Detach_AfterCertification_AppendsNoKeys) {
  // Certification turns marking off but leaves the cycle open until endCycle,
  // which requires an empty barrier worklist. The root scan already covered
  // A's keys.
  auto rtA = attach();
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{*rtA, &lv};
    ASSERT_NE(nullptr, boxedArray(*rtA, lv.box));
  }
  gc->beginCycle();
  traceToCertification();
  EXPECT_FALSE(gc->markingOn());

  rtA.reset();
  EXPECT_EQ(0u, gc->mutatorCountSlow()) << "A did not detach";
  EXPECT_TRUE(gc->barrierWorklistSlow().empty())
      << "keys appended after certification would never be traced";
  gc->endCycle();
  gc->sweep();
  EXPECT_EQ(1u, gc->liveCellCountSlow()) << "the root scan kept A's cell";
}

TEST_F(SharedAttachTest, Prune_OfMoreKeysThanARingHolds_KeepsEveryCell) {
  // One dead key more than a ring holds, so the prune spills, before it takes
  // A's map lock, and every cell must still survive the cycle.
  constexpr uint32_t kCount = SharedBarrierRing::kCapacity + 1;
  auto rtA = attach();
  auto rtB = Runtime::create(kTestRTConfig);
  Runtime &a = *rtA;
  SharedMutator mutatorB{*gc, *rtB};

  // Every box is kept until the last is made: a local GC during the loop
  // would otherwise let the next makeBoxed prune keys before the cycle, with
  // nothing recorded. Then they all die, and their keys stay until the lookup
  // below, the only thing here that prunes.
  std::vector<SharedArray *> cells;
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
      PinnedValue<JSArray> boxes;
    } lv;
    LocalsRAII lraii{a, &lv};
    auto created = JSArray::create(a, kCount, 0);
    ASSERT_NE(ExecutionStatus::EXCEPTION, created.getStatus());
    lv.boxes = std::move(*created);
    for (uint32_t i = 0; i < kCount; ++i) {
      SharedArray *cell = boxedArray(a, lv.box);
      ASSERT_NE(nullptr, cell);
      ASSERT_NE(
          ExecutionStatus::EXCEPTION,
          JSArray::setElementAt(lv.boxes, a, i, lv.box));
      cells.push_back(cell);
    }
  }

  mutatorB.enterBarrier();
  gc->beginCycle();
  EXPECT_FALSE(gc->pollInFlightBarriers()) << "B's window holds the wait";
  SharedArray *holder = array(kCount);
  for (uint32_t i = 0; i < kCount; ++i)
    holder->initField(i, SharedValue::encodePointer(cells[i]));
  box(*rtB, holder);

  a.collect("test");
  EXPECT_EQ(nullptr, a.findBoxForSharedCell(cells[0]));
  EXPECT_EQ(0u, a.sharedObjectBoxCountSlow()) << "the lookup did not prune";
  EXPECT_EQ(SharedBarrierRing::kCapacity, gc->barrierWorklistSlow().size())
      << "the prune did not spill";
  mutatorB.exitBarrier();

  traceToCertification();
  gc->endCycle();
  gc->sweep();

  ASSERT_EQ(kCount + 1, gc->liveCellCountSlow())
      << "a cell whose key was pruned during the wait was swept";
  for (uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(cells[i], holder->at(i).getPointer());
}

TEST_F(SharedAttachTest, Prune_WithMarkingOff_ErasesWithoutPushing) {
  // The common case: an attached runtime prunes between cycles. A push needs
  // a stamp that says marking is on, so this prune must push nothing.
  auto rtA = attach();
  Runtime &a = *rtA;
  SharedArray *cell = nullptr;
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{a, &lv};
    cell = boxedArray(a, lv.box);
    ASSERT_NE(nullptr, cell);
  }
  a.collect("test");
  EXPECT_EQ(nullptr, a.findBoxForSharedCell(cell));
  EXPECT_EQ(0u, a.sharedObjectBoxCountSlow());
}

} // namespace
