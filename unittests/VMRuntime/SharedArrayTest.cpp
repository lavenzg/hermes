/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/HeapAlign.h"
#include "hermes/VM/SharedHeap/SharedArray.h"

#include <cstdint>

namespace {

using namespace hermes;
using namespace hermes::vm;

/// Ample enough that exhaustion is never the thing under test.
constexpr size_t kAmpleBudget = 8 << 20;

/// Free everything. Two sweeps are needed, not one: cells are born marked, so
/// the first only clears the birth mark and the second collects.
void drain(SharedGC &gc) {
  gc.sweep();
  gc.sweep();
}

struct SharedArrayTest : public ::testing::Test {
  SharedGC gc{kAmpleBudget};

  ~SharedArrayTest() override {
    // The destructor asserts every run has been swept.
    drain(gc);
  }
};

TEST_F(SharedArrayTest, Create_SetsKindAndLength) {
  auto *arr = SharedArray::create(gc, 4);
  ASSERT_NE(nullptr, arr);

  EXPECT_EQ(CellKind::SharedArrayKind, arr->getKind());
  EXPECT_EQ(4u, arr->length());
#ifndef NDEBUG
  // makeA stamps the kind and re-runs SharedGCCell's initialisers via the
  // placement-new, so a created array is valid where allocRaw's storage is not.
  EXPECT_TRUE(arr->isValid());
#endif
  EXPECT_TRUE(SharedArray::classof(arr));
}

TEST_F(SharedArrayTest, Create_EmptiesEveryElement) {
  auto *arr = SharedArray::create(gc, 8);
  ASSERT_NE(nullptr, arr);

  for (uint32_t i = 0; i < arr->length(); ++i)
    EXPECT_TRUE(arr->at(i).isEmpty()) << "element " << i;
}

TEST_F(SharedArrayTest, Create_EmptiesASlotItsPredecessorLeftDirty) {
  // The property that matters: allocation recycles slots, so an element left
  // unwritten holds whatever the previous occupant put there -- and a tracer
  // would follow it as a pointer.
  constexpr uint32_t kLength = 4;
  auto *survivor = SharedArray::create(gc, kLength);
  auto *dirty = SharedArray::create(gc, kLength);
  ASSERT_NE(nullptr, survivor);
  ASSERT_NE(nullptr, dirty);
  for (uint32_t i = 0; i < kLength; ++i)
    dirty->set(i, SharedValue::encodePointer(survivor));

  // Free `dirty` but keep its segment, so the next array of this size bucket
  // reuses the slot instead of landing on a fresh mapping.
  gc.sweep(); // clears the birth marks
  SharedGC::setCellMarkBit(survivor);
  gc.sweep();
  ASSERT_EQ(1u, gc.liveCellCountSlow());
  const size_t regionsBefore = gc.regionCountSlow();

  auto *fresh = SharedArray::create(gc, kLength);
  ASSERT_NE(nullptr, fresh);
  ASSERT_EQ(regionsBefore, gc.regionCountSlow()) << "wanted a recycled slot";
  for (uint32_t i = 0; i < fresh->length(); ++i)
    EXPECT_TRUE(fresh->at(i).isEmpty()) << "element " << i;
}

TEST_F(SharedArrayTest, SetAndGet_RoundTripAPointer) {
  auto *target = SharedArray::create(gc, 1);
  auto *arr = SharedArray::create(gc, 3);
  ASSERT_NE(nullptr, target);
  ASSERT_NE(nullptr, arr);

  arr->set(1, SharedValue::encodePointer(target));

  EXPECT_TRUE(arr->at(0).isEmpty());
  EXPECT_FALSE(arr->at(1).isEmpty());
  EXPECT_EQ(target, arr->at(1).getPointer());
  EXPECT_TRUE(arr->at(2).isEmpty());
}

TEST_F(SharedArrayTest, ZeroLength_IsAllocatableAndHasNoElements) {
  auto *arr = SharedArray::create(gc, 0);
  ASSERT_NE(nullptr, arr);

  EXPECT_EQ(0u, arr->length());
  EXPECT_EQ(arr->begin(), arr->end());
}

TEST_F(SharedArrayTest, Elements_AreContiguousAfterTheHeader) {
  // What lets a tracer walk length() slots from begin() with a fixed stride.
  constexpr uint32_t kLength = 5;
  auto *arr = SharedArray::create(gc, kLength);
  ASSERT_NE(nullptr, arr);

  EXPECT_EQ(
      reinterpret_cast<char *>(arr) + sizeof(SharedArray),
      reinterpret_cast<char *>(arr->begin()));
  EXPECT_EQ(kLength, static_cast<uint32_t>(arr->end() - arr->begin()));
  for (uint32_t i = 1; i < kLength; ++i)
    EXPECT_EQ(
        sizeof(SharedValue),
        static_cast<size_t>(
            reinterpret_cast<char *>(arr->begin() + i) -
            reinterpret_cast<char *>(arr->begin() + i - 1)));
}

TEST_F(SharedArrayTest, Create_GivesTheCellRoomForItsElements) {
  for (uint32_t n : {0u, 1u, 5u, 64u, 1000u}) {
    auto *arr = SharedArray::create(gc, n);
    ASSERT_NE(nullptr, arr) << "length " << n;
    EXPECT_GE(
        SharedArray::allocationSize(n),
        sizeof(SharedArray) + n * sizeof(SharedValue))
        << "length " << n;
    EXPECT_GE(SharedGC::allocatedSize(arr), SharedArray::allocationSize(n))
        << "length " << n;
    // Touching the last element must stay inside the cell. ASan poisons the
    // slack past a slot, so this is a real bounds check.
    if (n)
      arr->set(n - 1, SharedValue::encodePointer(arr));
  }
}

TEST_F(SharedArrayTest, Create_PastTheLargestBucketGetsItsOwnRun) {
  const uint32_t length =
      SharedGC::SizeBuckets::kMaxBucketSize / sizeof(SharedValue) + 16;
  ASSERT_GT(
      SharedArray::allocationSize(length),
      SharedGC::SizeBuckets::kMaxBucketSize);

  auto *arr = SharedArray::create(gc, length);
  ASSERT_NE(nullptr, arr);
  EXPECT_EQ(length, arr->length());
  for (uint32_t i = 0; i < length; ++i)
    ASSERT_TRUE(arr->at(i).isEmpty()) << "element " << i;
}

TEST_F(SharedArrayTest, Create_ReturnsNullWhenTheArrayCannotBeAllocated) {
  // An oversized array needs a run of its own, which this budget cannot cover.
  SharedGC small{SharedGC::kSegmentSize};
  EXPECT_EQ(
      nullptr,
      SharedArray::create(small, SharedGC::kSegmentSize / sizeof(SharedValue)));
  drain(small);

  // No cell can hold these. Computed in 32 bits, the size of 2^29 elements
  // would wrap to the header's alone, and the constructor would then empty
  // 4 GiB past the cell.
  const uint32_t longest =
      (SharedGC::maxCellSize() - sizeof(SharedArray)) / sizeof(SharedValue);
  EXPECT_LE(SharedArray::allocationSize(longest), SharedGC::maxCellSize());
  for (uint32_t length : {longest + 1, 1u << 29, UINT32_MAX}) {
    EXPECT_EQ(UINT32_MAX, SharedArray::allocationSize(length))
        << "length " << length;
    EXPECT_EQ(nullptr, SharedArray::create(gc, length)) << "length " << length;
  }
}

TEST_F(SharedArrayTest, Header_LeavesNoWastedSpaceBeforeTheElements) {
  // A shared slot is 8 bytes and must be 8-aligned, so the header has room for
  // the kind and the length with nothing left over. If this ever grows, every
  // array in the heap pays for it.
#ifdef NDEBUG
  EXPECT_EQ(8u, sizeof(SharedArray));
#endif
  EXPECT_EQ(0u, sizeof(SharedArray) % alignof(SharedValue));
}

} // namespace
