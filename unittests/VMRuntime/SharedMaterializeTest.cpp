/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/BoxedSharedObject.h"
#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedBarrierRing.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedGCBoxing-inline.h"
#include "hermes/VM/SharedHeap/SharedMutator.h"

#include "VMRuntimeTestHelpers.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace {

using namespace hermes;
using namespace hermes::vm;

constexpr size_t kAmpleBudget = 8 << 20;

/// Runs whatever a test sets at each of the materialiser's steps. A test
/// passes it to materializeSharedSlot as the hook.
struct StepHooks {
  std::function<void()> beforeWindowStep;
  std::function<void()> afterEntryStep;
  std::function<void()> afterTestStep;
  std::function<void()> afterReadStep;
  std::function<void()> beforeExitStep;

  void operator()(Runtime::MaterializeStep step) const {
    const std::function<void()> *action = nullptr;
    switch (step) {
      case Runtime::MaterializeStep::BeforeWindow:
        action = &beforeWindowStep;
        break;
      case Runtime::MaterializeStep::AfterEntry:
        action = &afterEntryStep;
        break;
      case Runtime::MaterializeStep::AfterTest:
        action = &afterTestStep;
        break;
      case Runtime::MaterializeStep::AfterRead:
        action = &afterReadStep;
        break;
      case Runtime::MaterializeStep::BeforeExit:
        action = &beforeExitStep;
        break;
    }
    if (*action)
      (*action)();
  }
};

/// A shared heap, and runtimes attached to it through Runtime::create. A test
/// that needs another runtime to store, or to hold a barrier window open, adds
/// a mutator built by hand, for a runtime created without the heap, since an
/// attached runtime's own mutator is private.
struct SharedMaterializeTest : public ::testing::Test {
  std::shared_ptr<SharedGC> gc{std::make_shared<SharedGC>(kAmpleBudget)};

  std::shared_ptr<Runtime> attach() {
    return Runtime::create(kTestRTConfig, gc);
  }

  ~SharedMaterializeTest() override {
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

  /// \return a new one-slot array holding \p cell, boxed by no runtime.
  SharedArray *holding(SharedGCCell *cell) {
    SharedArray *holder = array(1);
    if (holder)
      holder->initField(0, SharedValue::encodePointer(cell));
    return holder;
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

  /// Turn marking on and finish both in-flight waits, with no window open.
  void beginMarking() {
    gc->beginCycle();
    EXPECT_TRUE(gc->pollInFlightBarriers()) << "no window is open";
    gc->beginSecondInFlightPass();
    EXPECT_TRUE(gc->pollInFlightBarriers()) << "no window is open";
  }

  /// Scan the roots and trace until a round certifies.
  /// \pre both in-flight waits are done.
  void markToCertification() {
    gc->markRuntimeRoots();
    do {
      gc->drainMarkWorklist();
    } while (!gc->certifyRound());
  }

  /// What collect() does once its first wait is over, up to the round that
  /// certifies.
  void traceToCertification() {
    EXPECT_TRUE(gc->pollInFlightBarriers());
    gc->beginSecondInFlightPass();
    EXPECT_TRUE(gc->pollInFlightBarriers()) << "no window is open by now";
    markToCertification();
  }

  /// End a certified cycle and sweep, then check that no key outlived its
  /// cell.
  void endAndSweep() {
    gc->endCycle();
    gc->sweep();
#ifdef HERMES_SLOW_DEBUG
    EXPECT_EQ(0u, gc->countDanglingKeys());
#endif
  }
};

TEST_F(SharedMaterializeTest, Materialize_AnEmptySlot_GivesNoBox) {
  auto rtA = attach();
  SharedArray *holder = array(1);
  ASSERT_NE(nullptr, holder);

  EXPECT_FALSE(rtA->materializeSharedSlot(*holder->begin()));
  EXPECT_EQ(0u, rtA->sharedObjectBoxCountSlow());
}

TEST_F(SharedMaterializeTest, Materialize_BoxesTheCellAndRecordsIt) {
  auto rtA = attach();
  Runtime &a = *rtA;
  SharedArray *cell = array(1);
  SharedArray *holder = holding(cell);
  ASSERT_NE(nullptr, holder);

  struct : public Locals {
    PinnedValue<BoxedSharedObject> box;
  } lv;
  LocalsRAII lraii{a, &lv};
  lv.box = a.materializeSharedSlot(*holder->begin());
  ASSERT_TRUE(*lv.box);
  EXPECT_EQ(cell, lv.box->getSharedCell());
  EXPECT_EQ(1u, a.sharedObjectBoxCountSlow());
  EXPECT_EQ(*lv.box, a.findBoxForSharedCell(cell));
}

TEST_F(SharedMaterializeTest, Materialize_Again_GivesTheSameBox) {
  // Strict equality on a shared object compares boxes, so a second
  // materialisation must find the first box, even after a local GC moved it.
  auto rtA = attach();
  Runtime &a = *rtA;
  SharedArray *cell = array(1);
  SharedArray *holder = holding(cell);
  ASSERT_NE(nullptr, holder);

  struct : public Locals {
    PinnedValue<BoxedSharedObject> first;
    PinnedValue<BoxedSharedObject> second;
  } lv;
  LocalsRAII lraii{a, &lv};
  lv.first = a.materializeSharedSlot(*holder->begin());
  ASSERT_TRUE(*lv.first);
  a.collect("test");
  lv.second = a.materializeSharedSlot(*holder->begin());
  EXPECT_EQ(*lv.first, *lv.second);
  EXPECT_EQ(1u, a.sharedObjectBoxCountSlow());
}

TEST_F(SharedMaterializeTest, Materialize_WithMarkingOn_PushesOnAMissOnly) {
  // A hit needs no push: its key was scanned, or was pushed or born marked
  // when it was published.
  auto rtA = attach();
  Runtime &a = *rtA;
  SharedArray *cell = array(1);
  SharedArray *holder = holding(cell);
  ASSERT_NE(nullptr, holder);

  beginMarking();
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{a, &lv};
    lv.box = a.materializeSharedSlot(*holder->begin());
    EXPECT_TRUE(*lv.box);
    EXPECT_EQ(1u, gc->stealBarrierEntries()) << "the miss pushed nothing";
    EXPECT_EQ(std::vector<SharedGCCell *>{cell}, gc->barrierWorklistSlow());

    lv.box = a.materializeSharedSlot(*holder->begin());
    EXPECT_TRUE(*lv.box);
    EXPECT_EQ(0u, gc->stealBarrierEntries()) << "the hit pushed";
  }
  markToCertification();
  endAndSweep();
  // The holder is boxed by nothing.
  EXPECT_EQ(1u, gc->liveCellCountSlow());
}

TEST_F(SharedMaterializeTest, Materialize_ReadsTheSlotOnlyAfterTestingTheFlag) {
  // A read made before the test could return a value overwritten by a store
  // whose barrier tested the flag before marking turned on. Here marking turns
  // on between A's test and its read, and C then stores into the slot. The box
  // says which value A read, and the pushes say which side of the turn A's
  // test fell on: C's barrier pushes the cell it overwrites, and A, whose test
  // saw marking off, pushes nothing.
  auto rtA = attach();
  Runtime &a = *rtA;
  auto rtC = Runtime::create(kTestRTConfig);
  SharedMutator mutatorC{*gc, *rtC};
  SharedArray *before = array(1);
  SharedArray *after = array(1);
  SharedArray *holder = holding(before);
  ASSERT_NE(nullptr, holder);

  // EXPECT, not ASSERT, until the cycle ends.
  StepHooks hooks;
  hooks.afterTestStep = [&] {
    gc->beginCycle();
    holder->set(0, SharedValue::encodePointer(after), mutatorC);
  };
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{a, &lv};
    lv.box = a.materializeSharedSlot(*holder->begin(), hooks);
    EXPECT_TRUE(*lv.box && lv.box->getSharedCell() == after)
        << "the slot was read before the flag was tested";
  }
  EXPECT_EQ(1u, gc->stealBarrierEntries())
      << "only C's barrier should push: A tested before marking turned on";
  EXPECT_EQ(std::vector<SharedGCCell *>{before}, gc->barrierWorklistSlow());

  traceToCertification();
  endAndSweep();
}

TEST_F(SharedMaterializeTest, Materialize_TestsTheFlagOnlyAfterEntering) {
  // The in-flight check relies on each window's entry coming before its test:
  // then either the test sees marking on, or the collector's look finds the
  // window open. Here marking turns on between A's entry and its test, so the
  // test must see it on, and A pushes the cell it reads.
  auto rtA = attach();
  Runtime &a = *rtA;
  SharedArray *cell = array(1);
  SharedArray *holder = holding(cell);
  ASSERT_NE(nullptr, holder);

  StepHooks hooks;
  hooks.afterEntryStep = [&] {
    gc->beginCycle();
    EXPECT_EQ(1u, gc->awaitedCountSlow()) << "A's window is open";
  };
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{a, &lv};
    lv.box = a.materializeSharedSlot(*holder->begin(), hooks);
    EXPECT_TRUE(*lv.box);
  }
  EXPECT_EQ(1u, gc->stealBarrierEntries())
      << "A tested the flag before entering its window";

  traceToCertification();
  endAndSweep();
}

TEST_F(
    SharedMaterializeTest,
    Materialize_TheSlotChangesBeforeTheWindow_BoxesTheNewCell) {
  // The hit check's read is only a key. A whole cycle can pass before the
  // window, so the cell published is the one read inside it.
  auto rtA = attach();
  Runtime &a = *rtA;
  auto rtC = Runtime::create(kTestRTConfig);
  SharedMutator mutatorC{*gc, *rtC};
  SharedArray *before = array(1);
  SharedArray *after = array(1);
  SharedArray *holder = holding(before);
  ASSERT_NE(nullptr, holder);

  StepHooks hooks;
  hooks.beforeWindowStep = [&] {
    holder->set(0, SharedValue::encodePointer(after), mutatorC);
  };
  struct : public Locals {
    PinnedValue<BoxedSharedObject> box;
  } lv;
  LocalsRAII lraii{a, &lv};
  lv.box = a.materializeSharedSlot(*holder->begin(), hooks);
  ASSERT_TRUE(*lv.box);
  EXPECT_EQ(after, lv.box->getSharedCell());
  EXPECT_EQ(nullptr, a.findBoxForSharedCell(before));
  EXPECT_EQ(1u, a.sharedObjectBoxCountSlow());
}

TEST_F(
    SharedMaterializeTest,
    Materialize_TheSlotEmptiesBeforeTheWindow_GivesNoBox) {
  auto rtA = attach();
  Runtime &a = *rtA;
  auto rtC = Runtime::create(kTestRTConfig);
  SharedMutator mutatorC{*gc, *rtC};
  SharedArray *cell = array(1);
  SharedArray *holder = holding(cell);
  ASSERT_NE(nullptr, holder);

  StepHooks hooks;
  hooks.beforeWindowStep = [&] {
    holder->set(0, SharedValue::empty(), mutatorC);
  };
  EXPECT_FALSE(a.materializeSharedSlot(*holder->begin(), hooks));
  EXPECT_EQ(0u, a.sharedObjectBoxCountSlow()) << "the unused box was recorded";
}

TEST_F(
    SharedMaterializeTest,
    Materialize_TheSlotChangesToABoxedCell_GivesItsBox) {
  // The lookup inside the window can hit even though the hit check missed:
  // the slot now holds a cell this runtime already boxes. The new box is
  // dropped, and the cell keeps one box.
  auto rtA = attach();
  Runtime &a = *rtA;
  auto rtC = Runtime::create(kTestRTConfig);
  SharedMutator mutatorC{*gc, *rtC};
  SharedArray *unboxed = array(1);
  SharedArray *holder = holding(unboxed);
  ASSERT_NE(nullptr, holder);

  struct : public Locals {
    PinnedValue<BoxedSharedObject> existing;
    PinnedValue<BoxedSharedObject> box;
  } lv;
  LocalsRAII lraii{a, &lv};
  SharedArray *boxed = boxedArray(a, lv.existing);
  ASSERT_NE(nullptr, boxed);

  StepHooks hooks;
  hooks.beforeWindowStep = [&] {
    holder->set(0, SharedValue::encodePointer(boxed), mutatorC);
  };
  lv.box = a.materializeSharedSlot(*holder->begin(), hooks);
  EXPECT_EQ(*lv.existing, *lv.box);
  EXPECT_EQ(1u, a.sharedObjectBoxCountSlow());
}

TEST_F(SharedMaterializeTest, Materialize_ALocalGCBeforeTheWindow_KeepsTheBox) {
  // Making the box can run a local GC, which can move it, so the box must be
  // rooted from its allocation to its publish.
  auto rtA = attach();
  Runtime &a = *rtA;
  SharedArray *cell = array(1);
  SharedArray *holder = holding(cell);
  ASSERT_NE(nullptr, holder);

  StepHooks hooks;
  hooks.beforeWindowStep = [&] { a.collect("test"); };
  struct : public Locals {
    PinnedValue<BoxedSharedObject> box;
  } lv;
  LocalsRAII lraii{a, &lv};
  lv.box = a.materializeSharedSlot(*holder->begin(), hooks);
  ASSERT_TRUE(*lv.box);
  EXPECT_EQ(cell, lv.box->getSharedCell());
  EXPECT_EQ(*lv.box, a.findBoxForSharedCell(cell));
}

TEST_F(
    SharedMaterializeTest,
    Materialize_PrunesBeforeTheWindow_AndPushesBothKeys) {
  // A local GC while the box is made clears a dead box's entry. The prune runs
  // before the window opens, not from the lookup inside it, and pushes the
  // dead key; the materialiser pushes its cell.
  auto rtA = attach();
  Runtime &a = *rtA;
  SharedArray *dying = array(1);
  SharedArray *dyingHolder = holding(dying);
  SharedArray *cell = array(1);
  SharedArray *holder = holding(cell);
  ASSERT_NE(nullptr, dyingHolder);
  ASSERT_NE(nullptr, holder);
  // Boxed and dropped at once: only the key remains.
  EXPECT_TRUE(a.materializeSharedSlot(*dyingHolder->begin()));

  beginMarking();
  StepHooks hooks;
  hooks.beforeWindowStep = [&] { a.collect("test"); };
  size_t entriesInTheWindow = SIZE_MAX;
  hooks.afterEntryStep = [&] {
    entriesInTheWindow = a.sharedObjectBoxCountSlow();
  };
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{a, &lv};
    lv.box = a.materializeSharedSlot(*holder->begin(), hooks);
    EXPECT_TRUE(*lv.box);
    EXPECT_EQ(0u, entriesInTheWindow) << "the window opened before the prune";
    EXPECT_EQ(1u, a.sharedObjectBoxCountSlow()) << "the dead key was kept";
    EXPECT_EQ(2u, gc->stealBarrierEntries());
  }
  markToCertification();
  endAndSweep();
  // `dying` is kept by its push this cycle; the holders by nothing.
  EXPECT_EQ(2u, gc->liveCellCountSlow());

  // A window left open would hold the next cycle's wait.
  gc->beginCycle();
  EXPECT_TRUE(gc->pollInFlightBarriers()) << "a window was left open";
  gc->abortCycle();
}

TEST_F(
    SharedMaterializeTest,
    Materialize_WhenMarkingTurnsOnInsideTheWindow_KeepsTheCell) {
  // `x`, which A boxes, holds the only edge to `y`. B's window tests the flag
  // while marking is off and reads `y`, so B pushes nothing. Marking then
  // turns on, and C, whose store barrier tested the flag before that, empties
  // the slot with nothing recorded. The trace finds the slot empty, so only
  // B's key can keep `y`, and it does because the collector waits for B's
  // window, which closes only once the key is published.
  auto rtA = attach();
  auto rtB = attach();
  auto rtC = Runtime::create(kTestRTConfig);
  SharedMutator mutatorC{*gc, *rtC};

  struct : public Locals {
    PinnedValue<BoxedSharedObject> x;
  } lvA;
  LocalsRAII lraiiA{*rtA, &lvA};
  SharedArray *x = boxedArray(*rtA, lvA.x);
  ASSERT_NE(nullptr, x);
  SharedArray *y = array(1);
  ASSERT_NE(nullptr, y);
  x->initField(0, SharedValue::encodePointer(y));

  // EXPECT, not ASSERT, until the cycle ends: returning early would leave
  // windows and the cycle open.
  StepHooks hooks;
  hooks.afterReadStep = [&] {
    gc->beginCycle();
    EXPECT_EQ(2u, gc->awaitedCountSlow()) << "B's and C's windows are open";
    // C's store lands now, recorded nowhere.
    x->begin()->initField(SharedValue::empty());
    mutatorC.exitBarrier();
    EXPECT_FALSE(gc->pollInFlightBarriers()) << "B's window holds the wait";
  };
  hooks.beforeExitStep = [&] {
    EXPECT_EQ(1u, rtB->sharedObjectBoxCountSlow()) << "B's key is not out yet";
    EXPECT_FALSE(gc->pollInFlightBarriers())
        << "B's window closed before its key was published";
  };
  struct : public Locals {
    PinnedValue<BoxedSharedObject> y;
  } lvB;
  LocalsRAII lraiiB{*rtB, &lvB};
  mutatorC.enterBarrier();
  lvB.y = rtB->materializeSharedSlot(*x->begin(), hooks);
  EXPECT_TRUE(*lvB.y && lvB.y->getSharedCell() == y);

  traceToCertification();
  EXPECT_TRUE(SharedGC::isMarked(y)) << "B's key was not scanned";
  endAndSweep();

  // Counted before `y` is read: a swept cell's slot is poisoned.
  ASSERT_EQ(2u, gc->liveCellCountSlow()) << "y was swept";
  EXPECT_EQ(1u, y->length());
}

TEST_F(
    SharedMaterializeTest,
    Materialize_DuringTheWait_KeepsACellALateStoreUnlinks) {
  // As above, but B's window opens while the collector waits for C's, so its
  // test sees marking on: B pushes `y` as well as publishing its key, and C's
  // store then removes the only edge to `y`, recorded nowhere.
  auto rtA = attach();
  auto rtB = attach();
  auto rtC = Runtime::create(kTestRTConfig);
  SharedMutator mutatorC{*gc, *rtC};

  struct : public Locals {
    PinnedValue<BoxedSharedObject> x;
  } lvA;
  LocalsRAII lraiiA{*rtA, &lvA};
  SharedArray *x = boxedArray(*rtA, lvA.x);
  ASSERT_NE(nullptr, x);
  SharedArray *y = array(1);
  ASSERT_NE(nullptr, y);
  x->initField(0, SharedValue::encodePointer(y));

  mutatorC.enterBarrier();
  gc->beginCycle();
  EXPECT_FALSE(gc->pollInFlightBarriers()) << "C's window holds the wait";
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> y;
    } lvB;
    LocalsRAII lraiiB{*rtB, &lvB};
    lvB.y = rtB->materializeSharedSlot(*x->begin());
    EXPECT_TRUE(*lvB.y && lvB.y->getSharedCell() == y);
  }
  EXPECT_EQ(1u, gc->stealBarrierEntries()) << "B saw marking on but no push";
  x->begin()->initField(SharedValue::empty());
  mutatorC.exitBarrier();

  traceToCertification();
  EXPECT_TRUE(SharedGC::isMarked(y));
  endAndSweep();

  ASSERT_EQ(2u, gc->liveCellCountSlow()) << "y was swept";
  EXPECT_EQ(1u, y->length());
}

TEST_F(
    SharedMaterializeTest,
    Materialize_OfMoreCellsThanARingHolds_KeepsEveryCell) {
  // The last push spills inside the window, before the materialiser takes
  // A's map lock, and every cell must still survive the cycle.
  constexpr uint32_t kCount = SharedBarrierRing::kCapacity + 1;
  auto rtA = attach();
  Runtime &a = *rtA;
  SharedArray *holder = array(kCount);
  ASSERT_NE(nullptr, holder);
  std::vector<SharedArray *> cells;
  for (uint32_t i = 0; i < kCount; ++i) {
    cells.push_back(array(1));
    ASSERT_NE(nullptr, cells.back());
    holder->initField(i, SharedValue::encodePointer(cells.back()));
  }

  beginMarking();
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{a, &lv};
    for (uint32_t i = 0; i < kCount; ++i) {
      lv.box = a.materializeSharedSlot(holder->begin()[i]);
      EXPECT_TRUE(*lv.box);
    }
  }
  EXPECT_EQ(SharedBarrierRing::kCapacity, gc->barrierWorklistSlow().size())
      << "the materialiser's push did not spill";
  markToCertification();
  endAndSweep();

  // The holder is boxed by nothing, so only the cells remain.
  ASSERT_EQ(kCount, gc->liveCellCountSlow());
  for (SharedArray *cell : cells)
    EXPECT_EQ(1u, cell->length());
}

TEST_F(SharedMaterializeTest, Concurrent_TheBoxedCellIsSeenAsBuilt) {
  // A writer replaces the slot's cell with a newly built one after the
  // reader's hit check and before its window, so the read inside the window
  // is the reader's only acquire of the new cell. Nothing else orders the two
  // threads: they share no lock and no other atomic, each loads the other's
  // flag relaxed, and neither runs a gtest macro, since a failing one takes a
  // lock. SharedArray::create writes the array's kind and length with plain
  // stores, so weaken the read to relaxed and ThreadSanitizer flags a data
  // race on them. The writer still sets its flag with a release store, so
  // that on weakly ordered hardware the flag cannot become visible before the
  // slot store; loaded relaxed, it orders nothing.
  //
  // Both runtimes are created here, before the threads start: creating two
  // at once races inside libc's tzset, which ThreadSanitizer reports.
  constexpr uint32_t kBuiltLength = 3;
  auto rtReader = attach();
  auto rtWriter = Runtime::create(kTestRTConfig);
  SharedMutator writerMutator{*gc, *rtWriter};
  SharedArray *holder = holding(array(1));
  ASSERT_NE(nullptr, holder);
  std::atomic<bool> readerWaits{false};
  std::atomic<bool> stored{false};
  SharedGCCell *built = nullptr;
  SharedGCCell *boxed = nullptr;
  CellKind kind = CellKind::UninitializedKind;
  uint32_t length = 0;

  std::thread reader([&] {
    Runtime &rt = *rtReader;
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{rt, &lv};
    StepHooks hooks;
    hooks.beforeWindowStep = [&] {
      readerWaits.store(true, std::memory_order_relaxed);
      while (!stored.load(std::memory_order_relaxed))
        std::this_thread::yield();
    };
    lv.box = rt.materializeSharedSlot(*holder->begin(), hooks);
    if (!*lv.box)
      return;
    boxed = lv.box->getSharedCell();
    kind = boxed->getKind();
    length = static_cast<SharedArray *>(boxed)->length();
  });
  std::thread writer([&] {
    while (!readerWaits.load(std::memory_order_relaxed))
      std::this_thread::yield();
    built = SharedArray::create(*gc, kBuiltLength);
    holder->set(0, SharedValue::encodePointer(built), writerMutator);
    stored.store(true, std::memory_order_release);
  });
  writer.join();
  reader.join();

  ASSERT_NE(nullptr, built);
  EXPECT_EQ(built, boxed);
  EXPECT_EQ(CellKind::SharedArrayKind, kind);
  EXPECT_EQ(kBuiltLength, length);
}

} // namespace
