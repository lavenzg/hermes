/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/VM/GC.h"
#include "hermes/VM/HeapAlign.h"
#include "hermes/VM/SharedHeap/SharedGCBoxing-inline.h"

#include "VMRuntimeTestHelpers.h"
#include "gtest/gtest.h"

#include <memory>

using namespace hermes::vm;

namespace {

/// Ample enough that budget exhaustion is never the thing under test.
constexpr size_t kAmpleBudget = 8 << 20;

/// A shared cell of its own fields only. SharedArray does not exist yet at this
/// point, and reusing its kind is all makeA needs -- the kind is only checked
/// for being in the shared range. Two fields, so the argument pack is exercised
/// with more than one argument on its way through both layers.
class TestCell final : public SharedGCCell {
 public:
  uint64_t a;
  uint64_t b;

  TestCell(uint64_t a, uint64_t b) : a(a), b(b) {}

  static constexpr CellKind getCellKind() {
    return CellKind::SharedArrayKind;
  }
};

/// Free everything in \p gc. Two sweeps always suffice: cells are born marked,
/// so the first clears the mark on whatever survives and the second collects
/// it. ~SharedGC would free what is left too.
struct DrainGuard {
  SharedGC &gc;
  explicit DrainGuard(SharedGC &gc) : gc(gc) {}
  ~DrainGuard() {
    // Nothing to turn off first: these boxing tests never turn marking on.
    // A test that turned marking on and left it on would trip
    // sweep()'s !markingOn() assert, and should end its own cycle.
    gc.sweep();
    gc.sweep();
  }
};

/// A shared heap and a runtime attached to it, as makeBoxed requires.
/// Destroyed in reverse order, so the runtime detaches before the heap is
/// drained.
struct AttachedRuntime {
  std::shared_ptr<SharedGC> heap;
  SharedGC &gc{*heap};
  DrainGuard drain{gc};
  std::shared_ptr<Runtime> rt{Runtime::create(kTestRTConfig, heap)};
  Runtime &runtime{*rt};
  GCScope gcScope{runtime};

  explicit AttachedRuntime(size_t budget)
      : heap(std::make_shared<SharedGC>(budget)) {}
};

struct SharedGCBoxingTest : public ::testing::Test {
  AttachedRuntime ample{kAmpleBudget};
  SharedGC &gc{ample.gc};
  Runtime &runtime{ample.runtime};
  GCScope &gcScope{ample.gcScope};
};

TEST_F(SharedGCBoxingTest, MakeBoxed_ReturnsABoxBoundToTheNewCell) {
  auto box = runtime.makeHandle(
      gc.makeBoxed<TestCell>(runtime, sizeof(TestCell), 0x1234u, 0x5678u));
  ASSERT_TRUE(*box);

  auto *cell = box->getSharedCell();
  ASSERT_NE(nullptr, cell);
  EXPECT_EQ(CellKind::SharedArrayKind, cell->getKind());
  // Both arguments, so the pack really did cross makeBoxed and makeALocked.
  EXPECT_EQ(0x1234u, static_cast<TestCell *>(cell)->a);
  EXPECT_EQ(0x5678u, static_cast<TestCell *>(cell)->b);
}

TEST_F(SharedGCBoxingTest, MakeBoxed_PublishesThePairIntoTheRuntimesMap) {
  // The point of the whole operation. A cell that reaches the mutator without
  // an entry here is a cell the shared collector's root scan cannot see, and it
  // is swept out from under its own runtime.
  auto box = runtime.makeHandle(
      gc.makeBoxed<TestCell>(runtime, sizeof(TestCell), 1u, 0u));
  ASSERT_TRUE(*box);

  EXPECT_EQ(box.get(), runtime.findBoxForSharedCell(box->getSharedCell()));
}

TEST_F(SharedGCBoxingTest, MakeBoxed_GivesEachCellItsOwnBox) {
  auto first = runtime.makeHandle(
      gc.makeBoxed<TestCell>(runtime, sizeof(TestCell), 1u, 0u));
  auto second = runtime.makeHandle(
      gc.makeBoxed<TestCell>(runtime, sizeof(TestCell), 2u, 0u));
  ASSERT_TRUE(*first);
  ASSERT_TRUE(*second);

  EXPECT_NE(first.get(), second.get());
  EXPECT_NE(first->getSharedCell(), second->getSharedCell());
  // And each resolves to its own, so the map did not overwrite one with the
  // other.
  EXPECT_EQ(first.get(), runtime.findBoxForSharedCell(first->getSharedCell()));
  EXPECT_EQ(
      second.get(), runtime.findBoxForSharedCell(second->getSharedCell()));
}

TEST_F(SharedGCBoxingTest, MakeBoxed_RoundsAVariableSizedRequest) {
  // Deliberately unaligned, and checked through the slot the allocator actually
  // handed out -- asserting only that the fields round-trip would pass with the
  // rounding removed in any build where allocLocked's assert is compiled out.
  const uint32_t size = sizeof(TestCell) + 4;
  auto box =
      runtime.makeHandle(gc.makeBoxed<TestCell>(runtime, size, 0x5678u, 0u));
  ASSERT_TRUE(*box);

  auto *cell = box->getSharedCell();
  EXPECT_EQ(heapAlignSize(size), SharedGC::allocatedSize(cell));
  EXPECT_EQ(0x5678u, static_cast<TestCell *>(cell)->a);
}

TEST_F(SharedGCBoxingTest, MakeBoxed_RejectsAnOversizedRequest) {
  // Past what any region can hold, so the request is rejected before the box is
  // even allocated. allocLocked only asserts this, so letting it through is a
  // Release-build corruption rather than a clean failure.
  //
  // The empty box map at the end of this test cannot show that the rejection
  // came *before* the box: the budget-exhausted path leaves the map empty too,
  // and that one does allocate a box first. Watching the local heap is what
  // separates them, so it is what actually pins the ordering the header
  // advertises.
  //
  // totalAllocatedBytes rather than DebugHeapInfo::numAllocatedObjects: the
  // latter is only maintained by MallocGC, so under the default Hades build it
  // sits at zero and would make this assertion vacuous. This one is cumulative,
  // so a young-gen collection landing in the middle cannot lower it either.
  GCBase::HeapInfo heapInfo;
  runtime.getHeap().getHeapInfo(heapInfo);
  const uint64_t allocatedBefore = heapInfo.totalAllocatedBytes;

  EXPECT_FALSE(gc.makeBoxed<TestCell>(
      runtime, SharedGC::maxCellSize() + HeapAlign, 1u, 0u));

  runtime.getHeap().getHeapInfo(heapInfo);
  EXPECT_EQ(allocatedBefore, heapInfo.totalAllocatedBytes)
      << "the size check ran after the box was allocated";

  // And a size whose rounding overflows 32 bits must not wrap into a small one
  // and quietly allocate a cell far too short for its caller.
  EXPECT_FALSE(gc.makeBoxed<TestCell>(runtime, UINT32_MAX, 1u, 0u));

  EXPECT_EQ(0u, runtime.sharedObjectBoxCountSlow())
      << "a rejected request left an entry behind";
}

TEST_F(SharedGCBoxingTest, MakeBoxed_IsEmptyWhenTheBudgetIsExhausted) {
  // The other failure path: the request is servable in principle, so the box is
  // allocated, and only then does the cell allocation fail. The box is dropped
  // on the floor as garbage and nothing is published.
  AttachedRuntime small{SharedGC::kSegmentSize};

  auto first = small.runtime.makeHandle(
      small.gc.makeBoxed<TestCell>(small.runtime, sizeof(TestCell), 1u, 0u));
  EXPECT_TRUE(*first);
  EXPECT_EQ(1u, small.runtime.sharedObjectBoxCountSlow());

  // A second size bucket would need a second segment, which the budget refuses.
  EXPECT_FALSE(small.gc.makeBoxed<TestCell>(
      small.runtime, SharedGC::SizeBuckets::kMaxBucketSize, 2u, 0u));
  EXPECT_EQ(1u, small.runtime.sharedObjectBoxCountSlow())
      << "a failed allocation published something";
}

#ifndef NDEBUG
TEST_F(SharedGCBoxingTest, MakeBoxed_ConsumesNoHandleSlotOfTheCallers) {
  // It roots the box in its own Locals, not in whatever GCScope the caller
  // happens to have open. That matters on the failure path: an OOM-retry loop
  // calls this repeatedly, and a slot allocated in the caller's scope is
  // reclaimed only when that scope ends -- so each rejected attempt would pin
  // one more dead, unbound box.
  AttachedRuntime small{SharedGC::kSegmentSize};
  auto first = small.runtime.makeHandle(
      small.gc.makeBoxed<TestCell>(small.runtime, sizeof(TestCell), 1u, 0u));
  ASSERT_TRUE(*first);

  const size_t before = small.gcScope.getHandleCountDbg();
  for (int i = 0; i < 16; ++i) {
    // Servable in principle, so a box is allocated before the cell fails.
    EXPECT_FALSE(small.gc.makeBoxed<TestCell>(
        small.runtime, SharedGC::SizeBuckets::kMaxBucketSize, 2u, 0u));
  }
  EXPECT_EQ(before, small.gcScope.getHandleCountDbg())
      << "makeBoxed pinned a slot in the caller's scope";
}
#endif

TEST_F(SharedGCBoxingTest, MakeBoxed_PrunesWhatALocalGCFoundDead) {
  // A runtime that only allocates never looks anything up. If allocating did
  // not prune, every key it inserted would stay a root after its box died.
  struct : public Locals {
    PinnedValue<BoxedSharedObject> kept;
    PinnedValue<BoxedSharedObject> made;
  } lv;
  LocalsRAII lraii{runtime, &lv};
  lv.kept = gc.makeBoxed<TestCell>(runtime, sizeof(TestCell), 1u, 0u);
  ASSERT_TRUE(*lv.kept);
  // Rooted by nothing, so the collection finds this box dead.
  EXPECT_TRUE(gc.makeBoxed<TestCell>(runtime, sizeof(TestCell), 2u, 0u));
  runtime.collect("test");
  ASSERT_EQ(2u, runtime.sharedObjectBoxCountSlow())
      << "the collection erased an entry itself";

  lv.made = gc.makeBoxed<TestCell>(runtime, sizeof(TestCell), 3u, 0u);
  ASSERT_TRUE(*lv.made);
  EXPECT_EQ(2u, runtime.sharedObjectBoxCountSlow())
      << "the dead entry outlived the allocation, or a live one did not";
}

TEST_F(SharedGCBoxingTest, MakeBoxed_SurvivesALocalGC) {
  // The box moves; the raw cell pointer it carries must not be touched, and the
  // map must follow the box to where it moved.
  auto box = runtime.makeHandle(
      gc.makeBoxed<TestCell>(runtime, sizeof(TestCell), 0xABCDu, 0u));
  ASSERT_TRUE(*box);
  auto *cell = box->getSharedCell();
  [[maybe_unused]] auto *before = box.get();

  runtime.collect("test");

#if HERMESVM_GCKIND != _HERMESVM_GCVALUE_MALLOC
  // Hades evacuates the young generation, so a box allocated a moment ago
  // always moves. Without this the map check below would hold trivially.
  //
  // Hades only. MallocGC relocates a cell just under handle sanitization
  // (MallocGC.cpp's accept()); otherwise it marks in place and the box stays
  // put, so asserting movement there would fail on correct code.
  ASSERT_NE(before, box.get());
#endif
  EXPECT_EQ(cell, box->getSharedCell());
  EXPECT_EQ(0xABCDu, static_cast<TestCell *>(cell)->a);
  EXPECT_EQ(box.get(), runtime.findBoxForSharedCell(cell));
}

} // namespace
