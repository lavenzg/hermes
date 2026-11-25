/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/CellKind.h"
#include "hermes/VM/HeapAlign.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedGCCell.h"

#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace {

using namespace hermes;
using namespace hermes::vm;

/// Ample enough that budget exhaustion is never the thing under test.
constexpr size_t kAmpleBudget = 8 << 20;

/// Free everything. Two sweeps always suffice: cells are born marked, so the
/// first clears the mark on whatever survives and the second collects it.
void drain(SharedGC &gc) {
  gc.sweep();
  gc.sweep();
}

/// A cell that is only its own fields, for the fixed-size half of makeA.
class TestCell final : public SharedGCCell {
 public:
  uint64_t a;
  uint64_t b;

  TestCell(uint64_t a, uint64_t b) : a(a), b(b) {}

  static constexpr CellKind getCellKind() {
    return CellKind::SharedArrayKind;
  }
};

/// A cell with trailing data, for the variable-size half.
///
/// alignas is load-bearing because data() hand-rolls the offset:
/// sizeof(SharedGCCell) differs by build configuration, so without it
/// `this + 1` can land off an 8-byte boundary -- silent under ASan, a hard
/// abort under UBSan. A production cell should use llvh::TrailingObjects,
/// which aligns the offset for you; SharedArray does.
class alignas(uint64_t) TestVarCell final : public SharedGCCell {
 public:
  uint32_t count;

  explicit TestVarCell(uint32_t count) : count(count) {}

  static constexpr CellKind getCellKind() {
    return CellKind::SharedArrayKind;
  }

  static uint32_t sizeFor(uint32_t n) {
    return heapAlignSize(
        static_cast<gcheapsize_t>(sizeof(TestVarCell) + n * sizeof(uint64_t)));
  }

  uint64_t *data() {
    return reinterpret_cast<uint64_t *>(this + 1);
  }
};

/// makeA<T> is the only way to allocate: it holds mtx_ across T's constructor,
/// which is what keeps a slot that is allocated but not yet a cell invisible to
/// a concurrent sweeper. There is deliberately no raw-storage entry point to
/// build a test helper on.
struct SharedGCTest : public ::testing::Test {
  SharedGC gc{kAmpleBudget};

  ~SharedGCTest() override {
    drain(gc);
  }
};

using SB = SharedGC::SizeBuckets;

TEST(SharedGCSizeBucketTest, Strides_AreHeapAlignedAscendingAndSpanTheRange) {
  EXPECT_EQ(HeapAlign, SB::strideOf(0));
  EXPECT_EQ(SB::kMaxBucketSize, SB::strideOf(SB::kNumBuckets - 1));

  for (uint32_t c = 0; c < SB::kNumBuckets; ++c) {
    const uint32_t stride = SB::strideOf(c);
    EXPECT_EQ(0u, stride % HeapAlign) << "bucket " << c;
    if (c) {
      EXPECT_GT(stride, SB::strideOf(c - 1)) << "bucket " << c;
    }
  }
}

TEST(SharedGCSizeBucketTest, IndexOf_PicksTheSmallestBucketThatFits) {
  // Aligned sizes only: indexOf requires one, and makeA is what rounds a
  // caller's request up before it gets here.
  for (uint32_t size = HeapAlign; size <= SB::kMaxBucketSize;
       size += HeapAlign) {
    const uint32_t c = SB::indexOf(size);
    ASSERT_LT(c, SB::kNumBuckets) << "size " << size;
    EXPECT_GE(SB::strideOf(c), size) << "size " << size;
    if (c) {
      EXPECT_LT(SB::strideOf(c - 1), size) << "size " << size;
    }
  }
}

TEST(SharedGCSizeBucketTest, IndexOf_IsTheInverseOfStrideOf) {
  for (uint32_t c = 0; c < SB::kNumBuckets; ++c)
    EXPECT_EQ(c, SB::indexOf(SB::strideOf(c))) << "bucket " << c;
}

TEST(SharedGCSizeBucketTest, Groups_BoundRoundingWasteAtOneOverNGroupPlusOne) {
  // The first step of each group is the widest, so it sets the bound. Below
  // the cutoff the classes step by HeapAlign and the bound does not apply.
  for (uint32_t c = 1; c < SB::kNumBuckets; ++c) {
    const uint32_t stride = SB::strideOf(c);
    if (stride <= SB::kPreciseCutoff)
      continue;
    // Worst case: a request one byte above the previous class.
    const uint32_t wasted = stride - (SB::strideOf(c - 1) + 1);
    EXPECT_LE(wasted * (SB::kNumGroup + 1), stride)
        << "bucket " << c << " stride " << stride;
  }
}

TEST_F(SharedGCTest, MakeA_ConstructsAndPassesArgs) {
  auto *cell = gc.makeA<TestCell>(sizeof(TestCell), uint64_t{7}, uint64_t{11});
  ASSERT_NE(nullptr, cell);

  EXPECT_EQ(7u, cell->a);
  EXPECT_EQ(11u, cell->b);
  EXPECT_EQ(CellKind::SharedArrayKind, cell->getKind());
#ifndef NDEBUG
  // The placement-new writes SharedGCCell's debug magic. The allocator never
  // does, which is what keeps isValid() able to tell a constructed cell from
  // an unclaimed slot.
  EXPECT_TRUE(cell->isValid());
#endif
}

TEST_F(SharedGCTest, MakeA_FixedSizeOccupiesItsTypesBucket) {
  auto *cell = gc.makeA<TestCell>(sizeof(TestCell), uint64_t{0}, uint64_t{0});
  ASSERT_NE(nullptr, cell);

  // sizeof(TestCell) is below the precise cutoff, so its bucket is exact and
  // the stride is the size.
  EXPECT_EQ(heapAlignSize(sizeof(TestCell)), SharedGC::allocatedSize(cell));
}

TEST_F(SharedGCTest, MakeA_TrailingDataGetsTheSpaceItAskedFor) {
  // The variable-size half: one entry point serves both, so a caller that
  // computes a length-dependent total gets a cell that actually holds it.
  constexpr uint32_t kCount = 6;
  auto *cell = gc.makeA<TestVarCell>(TestVarCell::sizeFor(kCount), kCount);
  ASSERT_NE(nullptr, cell);

  EXPECT_EQ(kCount, cell->count);
  EXPECT_GE(SharedGC::allocatedSize(cell), TestVarCell::sizeFor(kCount));

  // Writing every trailing word must stay inside the cell; ASan would object
  // otherwise, since the slack past a slot is poisoned.
  for (uint32_t i = 0; i < kCount; ++i)
    cell->data()[i] = i + 1;
  for (uint32_t i = 0; i < kCount; ++i)
    EXPECT_EQ(i + 1, cell->data()[i]);
}

TEST_F(SharedGCTest, MakeA_RoundsAnUnalignedSizeUp) {
  // allocLocked asserts its size is heap-aligned, so without the rounding this
  // aborts an assert-enabled build. The exact expectation is what makes the
  // rounding observable at all: indexOf is itself rounding-tolerant, so a
  // bucket-level check would pass either way.
  const uint32_t unaligned = sizeof(TestVarCell) + 1;
  auto *cell = gc.makeA<TestVarCell>(unaligned, uint32_t{0});
  ASSERT_NE(nullptr, cell);
  EXPECT_EQ(
      heapAlignSize(static_cast<gcheapsize_t>(unaligned)),
      SharedGC::allocatedSize(cell));
}

TEST_F(SharedGCTest, MakeA_BornMarked) {
  auto *cell = gc.makeA<TestCell>(sizeof(TestCell), uint64_t{1}, uint64_t{2});
  ASSERT_NE(nullptr, cell);
  EXPECT_FALSE(gc.markingOn());
  EXPECT_TRUE(SharedGC::isMarked(cell));
}

TEST_F(SharedGCTest, MakeA_FillsPastOneSegmentWithoutOverlapping) {
  // One cell more than a segment holds, which drives allocSmallLocked's cursor
  // off the end of the list and into createRegion. Every cell still holding
  // what it was constructed with is what shows no slot was handed out twice.
  const size_t perSegment =
      (SharedGC::kSegmentSize - SharedGC::kPayloadOffset) / sizeof(TestCell);
  std::vector<TestCell *> cells;
  for (size_t i = 0; i <= perSegment; ++i) {
    auto *c = gc.makeA<TestCell>(sizeof(TestCell), i, i);
    ASSERT_NE(nullptr, c);
    cells.push_back(c);
  }

  EXPECT_GE(gc.regionCountSlow(), 2u);
  EXPECT_EQ(cells.size(), gc.liveCellCountSlow());
  for (size_t i = 0; i < cells.size(); ++i) {
    ASSERT_EQ(i, cells[i]->a) << "cell " << i;
    ASSERT_EQ(i, cells[i]->b) << "cell " << i;
  }
}

TEST_F(SharedGCTest, MakeA_OverMaxBucketSize_GetsItsOwnRegion) {
  // The oversized path, which the bucketed tests never reach: a region of
  // several segments in largeRegions_. Two segments of payload plus the header
  // need three.
  const uint32_t big = 2 * SharedGC::kSegmentSize;
  auto *cell = gc.makeA<TestVarCell>(big, uint32_t{0});
  ASSERT_NE(nullptr, cell);

  EXPECT_GE(SharedGC::allocatedSize(cell), big);
  EXPECT_EQ(3 * SharedGC::kSegmentSize, gc.footprint());
  // All three segments go back when the cell is freed.
  drain(gc);
  EXPECT_EQ(0u, gc.footprint());
}

TEST_F(SharedGCTest, MakeA_DuringASweep_ReusesSlotsTheSweepFreed) {
  // At the budget, an allocation between two regions of a sweep must take a
  // slot that sweep has freed: it rewinds the bucket's cursor only at the end,
  // and a new segment would be refused.
  SharedGC small{2 * SharedGC::kSegmentSize};
  // Two cells that the second sweep from here frees: the first clears the
  // marks they were born with.
  EXPECT_NE(nullptr, small.makeA<TestCell>(sizeof(TestCell), 1u, 1u));
  EXPECT_NE(nullptr, small.makeA<TestCell>(sizeof(TestCell), 2u, 2u));
  small.sweep();
  // Born marked, so they all survive the next sweep: the rest of the first
  // segment, then a second, until the budget refuses.
  while (small.makeA<TestCell>(sizeof(TestCell), 3u, 3u)) {
  }
  EXPECT_EQ(2 * SharedGC::kSegmentSize, small.footprint());

  bool tried = false;
  TestCell *during = nullptr;
  const std::function<void()> afterEachRegion = [&] {
    if (tried)
      return;
    tried = true;
    during = small.makeA<TestCell>(sizeof(TestCell), 4u, 4u);
  };
  small.sweep(afterEachRegion);
  EXPECT_TRUE(tried);
  EXPECT_NE(nullptr, during) << "refused with two freed slots in the heap";
  drain(small);
}

TEST_F(SharedGCTest, MakeA_OverMaxCellSize_ReturnsNull) {
  const size_t before = gc.footprint();
  // Just past the limit, so the range check is what rejects it. allocLocked
  // only asserts this, so letting one through corrupts a Release build instead
  // of failing cleanly.
  EXPECT_EQ(
      nullptr,
      gc.makeA<TestVarCell>(SharedGC::maxCellSize() + HeapAlign, uint32_t{0}));
  // And a size whose rounding overflows 32 bits. The rounding is done in 64
  // bits and compared before the truncation back to 32, so this is rejected
  // rather than wrapping to a 0 that allocLocked would take for a legitimate
  // request.
  EXPECT_EQ(
      nullptr,
      gc.makeA<TestVarCell>(std::numeric_limits<uint32_t>::max(), uint32_t{0}));
  EXPECT_EQ(before, gc.footprint());
}

TEST_F(SharedGCTest, MakeA_BudgetExhausted_ReturnsNull) {
  SharedGC small{SharedGC::kSegmentSize};
  // EXPECT rather than ASSERT, so the refusal below is checked even if this
  // fails.
  EXPECT_NE(
      nullptr,
      small.makeA<TestCell>(sizeof(TestCell), uint64_t{1}, uint64_t{2}));
  // A second size bucket would need a second segment, which the budget refuses.
  EXPECT_EQ(nullptr, small.makeA<TestVarCell>(SB::kMaxBucketSize, uint32_t{0}));
  drain(small);
}

TEST_F(SharedGCTest, MakeA_ObeysTheCollector) {
  auto *live = gc.makeA<TestCell>(sizeof(TestCell), uint64_t{3}, uint64_t{4});
  auto *dead = gc.makeA<TestCell>(sizeof(TestCell), uint64_t{5}, uint64_t{6});
  ASSERT_NE(nullptr, live);
  ASSERT_NE(nullptr, dead);

  ASSERT_EQ(0u, gc.sweep()); // clears the birth marks
  SharedGC::setCellMarkBit(live);
  gc.sweep();

  EXPECT_EQ(1u, gc.liveCellCountSlow());
  EXPECT_EQ(3u, live->a);
  EXPECT_EQ(4u, live->b);
}

#ifndef NDEBUG
TEST(SharedGCTeardownTest, Teardown_FreesCellsLeftInTheHeap) {
  // Cells in two size buckets and one oversized region, none of them swept.
  // No runtime is attached, so teardown finds them all unreachable and frees
  // them, and then checks that the heap is empty.
  auto heap = std::make_unique<SharedGC>(kAmpleBudget);
  EXPECT_NE(nullptr, heap->makeA<TestCell>(sizeof(TestCell), 1, 2));
  EXPECT_NE(nullptr, heap->makeA<TestVarCell>(TestVarCell::sizeFor(8), 8u));
  const uint32_t oversized =
      SharedGC::SizeBuckets::kMaxBucketSize / sizeof(uint64_t) + 1;
  EXPECT_NE(
      nullptr,
      heap->makeA<TestVarCell>(TestVarCell::sizeFor(oversized), oversized));
  EXPECT_EQ(3u, heap->liveCellCountSlow());
  heap.reset();
}

TEST_F(SharedGCTest, MakeA_StampsADistinctDebugId) {
  auto *first = gc.makeA<TestCell>(sizeof(TestCell), uint64_t{0}, uint64_t{0});
  auto *second = gc.makeA<TestCell>(sizeof(TestCell), uint64_t{0}, uint64_t{0});
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  // Stamped after construction; written before, the base's initialisers would
  // have reset both to zero.
  EXPECT_NE(first->getDebugAllocationId(), second->getDebugAllocationId());
}
#endif

} // namespace
