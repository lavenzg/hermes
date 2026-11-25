/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/HeapAlign.h"
#include "hermes/VM/SharedHeap/SharedGC.h"

namespace {

using namespace hermes;
using namespace hermes::vm;

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

} // namespace
