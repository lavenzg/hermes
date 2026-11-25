/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGC-inline.h"

#include <cstdint>
#include <vector>

namespace {

using namespace hermes;
using namespace hermes::vm;

/// Ample enough that budget exhaustion is never the thing under test.
constexpr size_t kAmpleBudget = 8 << 20;

/// Free everything. Two sweeps always suffice: cells are born marked, so the
/// first clears the mark on whatever survives and the second collects it.
static void freeAll(SharedGC &gc) {
  gc.sweep();
  gc.sweep();
}

/// Records what it is handed, and nothing else. Not the mark acceptor, which
/// marks and enqueues -- having a second one is what proves the visitor is
/// acceptor-generic rather than welded to the collector.
struct RecordingAcceptor {
  std::vector<SharedGCCell *> cells;

  void accept(SharedGCCell *cell) {
    cells.push_back(cell);
  }
};

struct SharedSlotVisitorTest : public ::testing::Test {
  SharedGC gc{kAmpleBudget};

  ~SharedSlotVisitorTest() override {
    freeAll(gc);
  }

  SharedArray *array(uint32_t length) {
    SharedArray *arr = SharedArray::create(gc, length);
    EXPECT_NE(nullptr, arr);
    return arr;
  }
};

TEST_F(SharedSlotVisitorTest, MarkCell_YieldsEveryPointerInAddressOrder) {
  // Pointers at 0 and 2 of four slots, so the exact count pins three separate
  // properties at once and none of them needs its own test:
  //  - empty slots are filtered, because decoding is the visitor's job and no
  //    acceptor should have to learn the encoding;
  //  - the scan continues past an empty slot rather than stopping at the first
  //    one, since the pointer at 2 follows the hole at 1;
  //  - the acceptor is taken by reference -- were it by value, the pushes would
  //    land in a copy and `rec` would come back empty.
  auto *a = array(1);
  auto *b = array(1);
  auto *arr = array(4);
  arr->initField(0, SharedValue::encodePointer(a));
  arr->initField(2, SharedValue::encodePointer(b));

  RecordingAcceptor rec;
  SharedGC::markCell(rec, arr);

  ASSERT_EQ(2u, rec.cells.size());
  EXPECT_EQ(a, rec.cells[0]);
  EXPECT_EQ(b, rec.cells[1]);
}

TEST_F(SharedSlotVisitorTest, MarkCell_EmptyArray_YieldsNothing) {
  RecordingAcceptor rec;
  SharedGC::markCell(rec, array(0));

  EXPECT_TRUE(rec.cells.empty());
}

TEST_F(SharedSlotVisitorTest, MarkCell_WalksLengthNotStride) {
  // A loop derived from the cell's extent rather than its length reads the
  // slack past end(). Neither ASan nor the recorded count catches that on a
  // fresh segment -- finishAlloc unpoisons the whole stride, and zeroed slack
  // decodes as empty and is filtered -- so the slack has to be dirty. Fill the
  // whole stride, free it, and let the shorter array recycle the slot.
  constexpr uint32_t kLength = 17;
  using SB = SharedGC::SizeBuckets;
  const uint32_t stride =
      SB::strideOf(SB::indexOf(SharedArray::allocationSize(kLength)));
  // allocationSize(0), not sizeof(SharedArray): TrailingObjects aligns the
  // element offset up to alignof(SharedValue), so sizeof understates the header
  // and would overstate the capacity by an element -- enough to push the filler
  // into the next bucket and quietly break the recycling this test needs.
  const uint32_t maxElems =
      (stride - SharedArray::allocationSize(0)) / sizeof(SharedValue);
  ASSERT_GT(maxElems, kLength) << "this length no longer leaves slack";

  // Same bucket as the array under test, and kept alive so freeing the filler
  // does not release the segment and hand back a zeroed one.
  auto *keep = array(kLength);
  auto *filler = array(maxElems);
  for (uint32_t i = 0; i < maxElems; ++i)
    filler->initField(i, SharedValue::encodePointer(keep));

  gc.sweep(); // clears the birth marks
  SharedGC::setCellMarkBit(keep);
  gc.sweep(); // frees the filler, leaving its elements behind in the slot

  auto *arr = array(kLength);
  // The premise, asserted rather than assumed: everything below only tests
  // anything if arr landed on the filler's slot. On a fresh slot the slack is
  // zeroed, decodes as empty, is filtered, and the test passes with the bug it
  // exists to catch still present.
  ASSERT_EQ(static_cast<void *>(filler), static_cast<void *>(arr))
      << "wanted the filler's recycled slot, so the slack is dirty";
  for (uint32_t i = 0; i < kLength; ++i)
    arr->initField(i, SharedValue::encodePointer(keep));

  RecordingAcceptor rec;
  SharedGC::markCell(rec, arr);

  EXPECT_EQ(kLength, rec.cells.size());
}

} // namespace
