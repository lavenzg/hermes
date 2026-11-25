/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGC.h"

#include <cstdint>

namespace {

using namespace hermes;
using namespace hermes::vm;

/// Ample enough that budget exhaustion is never the thing under test.
constexpr size_t kAmpleBudget = 8 << 20;

/// Free everything: one complete cycle with no roots marks nothing, so its
/// sweep frees the heap. A legitimate collection rather than a teardown
/// special case -- beginCycle's reset clears whatever an abandoned cycle left
/// behind, which a bare sweep would misread as garbage.
static void freeAll(SharedGC &gc) {
  gc.beginCycle();
  while (!gc.certifyRound())
    gc.drainMarkWorklist();
  gc.endCycle();
  gc.sweep();
}

struct SharedMarkAcceptorTest : public ::testing::Test {
  SharedGC gc{kAmpleBudget};

  ~SharedMarkAcceptorTest() override {
    freeAll(gc);
  }

  SharedArray *array(uint32_t length) {
    SharedArray *arr = SharedArray::create(gc, length);
    EXPECT_NE(nullptr, arr);
    return arr;
  }

  /// Build the graph before any cycle turns marking on, so no barrier is owed.
  static void link(SharedArray *from, uint32_t i, SharedGCCell *to) {
    from->initField(i, SharedValue::encodePointer(to));
  }

  /// Clear the birth marks, so a cycle starts from an all-white heap. Call at
  /// most once per test: sweep() also whitens survivors, so a second call would
  /// free the whole graph. The live count is asserted rather than sweep()'s
  /// return, which is 0 whenever the emptied segments stay mapped.
  void whiten() {
    const size_t live = gc.liveCellCountSlow();
    gc.sweep();
    ASSERT_EQ(live, gc.liveCellCountSlow());
  }

  /// Certify and finish. Every cycle here traces to completion, so it ends
  /// rather than aborts -- and only a certified cycle may be swept.
  void certifyAndEnd() {
    do {
      gc.drainMarkWorklist();
    } while (!gc.certifyRound());
    gc.endCycle();
  }

  /// One full cycle, in the order a driver must use: turn marking on, trace,
  /// certify, then sweep, all on one thread: marks are set with relaxed
  /// atomics, so program order is what makes them visible to sweep().
  void cycleFrom(SharedGCCell *root) {
    gc.beginCycle();
    gc.markRoots({root});
    gc.drainMarkWorklist();
    certifyAndEnd();
    gc.sweep();
  }
};

TEST_F(SharedMarkAcceptorTest, Mark_FollowsAChainTransitively) {
  auto *a = array(1);
  auto *b = array(1);
  auto *c = array(1);
  auto *unreached = array(1);
  link(a, 0, b);
  link(b, 0, c);
  whiten();

  gc.beginCycle();
  gc.markRoots({a});
  EXPECT_EQ(3u, gc.drainMarkWorklist());
  certifyAndEnd();

  EXPECT_TRUE(SharedGC::isMarked(a));
  EXPECT_TRUE(SharedGC::isMarked(b));
  EXPECT_TRUE(SharedGC::isMarked(c));
  EXPECT_FALSE(SharedGC::isMarked(unreached));
}

TEST_F(SharedMarkAcceptorTest, Mark_TerminatesOnACycle) {
  auto *a = array(2);
  auto *b = array(1);
  link(a, 0, b);
  link(a, 1, a);
  link(b, 0, a);
  whiten();

  gc.beginCycle();
  gc.markRoots({a});
  EXPECT_EQ(2u, gc.drainMarkWorklist());
  certifyAndEnd();
}

TEST_F(SharedMarkAcceptorTest, Mark_TracesEveryRoot) {
  auto *a = array(1);
  auto *b = array(1);
  auto *aChild = array(1);
  auto *unreached = array(1);
  link(a, 0, aChild);
  whiten();

  gc.beginCycle();
  gc.markRoots({a, b});
  EXPECT_EQ(3u, gc.drainMarkWorklist());
  certifyAndEnd();

  EXPECT_TRUE(SharedGC::isMarked(b));
  EXPECT_TRUE(SharedGC::isMarked(aChild));
  EXPECT_FALSE(SharedGC::isMarked(unreached));
}

TEST_F(SharedMarkAcceptorTest, Mark_SkipsEmptySlots) {
  // getPointer() asserts on an empty slot, so a marker that failed to filter
  // would trip it rather than mark something wrong.
  auto *a = array(4);
  auto *b = array(0);
  link(a, 2, b);
  whiten();

  gc.beginCycle();
  gc.markRoots({a});
  EXPECT_EQ(2u, gc.drainMarkWorklist());
  certifyAndEnd();
  EXPECT_TRUE(SharedGC::isMarked(b));
}

TEST_F(SharedMarkAcceptorTest, Mark_IsIdempotent) {
  auto *a = array(1);
  whiten();

  gc.beginCycle();
  // Re-marking the same root traces nothing the second time: the bit set at
  // push is what keeps a cell from being enqueued twice.
  gc.markRoots({a});
  EXPECT_EQ(1u, gc.drainMarkWorklist());
  gc.markRoots({a});
  EXPECT_EQ(0u, gc.drainMarkWorklist());
  certifyAndEnd();
}

TEST_F(SharedMarkAcceptorTest, Mark_NeverTracesABornMarkedCell) {
  // Black allocation, and also why "nothing new was marked" is not a
  // termination test on its own. Allocated inside the cycle: beginCycle's bulk
  // reset whitens anything older, so a cell is only black if it was born after
  // marking was turned on.
  gc.beginCycle();
  auto *a = array(1);
  auto *b = array(1);
  link(a, 0, b);

  gc.markRoots({a});
  EXPECT_EQ(0u, gc.drainMarkWorklist());
  certifyAndEnd();
}

TEST_F(SharedMarkAcceptorTest, Sweep_FreesWhatTheMarkerDidNotReach) {
  auto *head = array(1);
  auto *mid = array(1);
  auto *tail = array(1);
  link(head, 0, mid);
  link(mid, 0, tail);
  for (int i = 0; i < 5; ++i)
    array(3);
  whiten();

  cycleFrom(head);

  EXPECT_EQ(3u, gc.liveCellCountSlow());
  EXPECT_EQ(mid, head->at(0).getPointer());
  EXPECT_EQ(tail, mid->at(0).getPointer());
}

TEST_F(SharedMarkAcceptorTest, Mark_ThenSweep_SurvivesTwoCycles) {
  // Catches a marker that leans on birth marks. The second cycle needs no
  // whitening: cycle 1's sweep already cleared the survivors' marks.
  auto *head = array(1);
  auto *tail = array(1);
  link(head, 0, tail);
  array(1);
  whiten();

  cycleFrom(head);
  ASSERT_EQ(2u, gc.liveCellCountSlow());
  cycleFrom(head);

  EXPECT_EQ(2u, gc.liveCellCountSlow());
  EXPECT_EQ(tail, head->at(0).getPointer());
}

} // namespace
