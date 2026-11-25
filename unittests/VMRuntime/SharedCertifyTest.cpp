/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedMutator.h"

#include "VMRuntimeTestHelpers.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

using namespace hermes;
using namespace hermes::vm;

/// Ample enough that budget exhaustion is never the thing under test.
constexpr size_t kAmpleBudget = 8 << 20;
constexpr uint32_t kRingCapacity = SharedBarrierRing::kCapacity;

/// Free everything. Two sweeps always suffice: cells are born marked, so the
/// first clears the mark on whatever survives and the second collects it.
static void freeAll(SharedGC &gc) {
  gc.sweep();
  gc.sweep();
}

struct SharedCertifyTest : public RuntimeTestFixture {
  SharedGC gc{kAmpleBudget};

  ~SharedCertifyTest() override {
    freeAll(gc);
  }

  SharedArray *array(uint32_t length) {
    SharedArray *arr = SharedArray::create(gc, length);
    EXPECT_NE(nullptr, arr);
    return arr;
  }

  static void link(SharedArray *from, uint32_t i, SharedGCCell *to) {
    from->set(i, SharedValue::encodePointer(to));
  }

  /// Clear the birth marks, so a cycle starts from an all-white heap.
  void whiten() {
    const size_t live = gc.liveCellCountSlow();
    gc.sweep();
    ASSERT_EQ(live, gc.liveCellCountSlow());
  }

  /// The round loop a driver runs: trace what is known, then try to certify;
  /// a failed round has left more on the local worklist.
  /// \return the number of rounds it took.
  uint32_t runToCertification() {
    uint32_t rounds = 0;
    do {
      gc.drainMarkWorklist();
      ++rounds;
    } while (!gc.certifyRound());
    return rounds;
  }
};

TEST_F(SharedCertifyTest, Certify_TurnsMarkingOffSoTheCycleCanBeSwept) {
  gc.beginCycle();
  ASSERT_TRUE(gc.markingOn());
  EXPECT_TRUE(gc.certifyRound());
  EXPECT_FALSE(gc.markingOn())
      << "certification must turn marking off inside its hold";
  gc.endCycle();
  gc.sweep();
}

TEST_F(SharedCertifyTest, Certify_RecoversACellRecordedOnlyInARing) {
  // The safety property, made observable: nothing reaches `cell` except the
  // ring entry, so if certification fails to recover it the sweep frees it.
  auto *cell = array(1);
  whiten();
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  mutator.push(cell, gc.cycleSeq());
  EXPECT_FALSE(gc.certifyRound()) << "an unmarked cell is new work";
  EXPECT_TRUE(gc.markingOn()) << "a failed round must leave marking on";
  EXPECT_TRUE(SharedGC::isMarked(cell));

  gc.drainMarkWorklist();
  EXPECT_TRUE(gc.certifyRound());
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(1u, gc.liveCellCountSlow()) << "certification lost the cell";
}

TEST_F(SharedCertifyTest, Certify_ARingOfAlreadyMarkedCellsCertifies) {
  // The churn case, and why the measure is "newly marked" rather than "empty".
  // A runtime reassigning references among live cells refills its ring
  // forever; an emptiness test would never let the cycle end.
  auto *cell = array(1);
  whiten();
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  gc.markRoots({cell});
  gc.drainMarkWorklist();
  for (uint32_t i = 0; i < 32; ++i)
    mutator.push(cell, gc.cycleSeq());

  EXPECT_TRUE(gc.certifyRound()) << "already-marked churn is not new work";
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(1u, gc.liveCellCountSlow());
}

TEST_F(SharedCertifyTest, Certify_ConsumesEveryRingInOneRound) {
  // Distinct cells in distinct runtimes' rings, each reachable from nothing
  // else, so both assertions are load-bearing and one round has to visit every
  // registered ring rather than stopping at the first.
  auto *first = array(1);
  auto *second = array(1);
  whiten();
  SharedMutator one{gc, runtime};
  SharedMutator two{gc, runtime};

  gc.beginCycle();
  one.push(first, gc.cycleSeq());
  two.push(second, gc.cycleSeq());

  EXPECT_FALSE(gc.certifyRound());
  EXPECT_TRUE(SharedGC::isMarked(first)) << "the first runtime's ring";
  EXPECT_TRUE(SharedGC::isMarked(second)) << "the second runtime's ring";

  runToCertification();
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow());
}

TEST_F(SharedCertifyTest, Certify_MarksTheSpilledBacklogToo) {
  // The other half: cells that reached the barrier worklist by overflowing a
  // ring, rather than by still sitting in one. `spilled` is only ever in the
  // worklist and `inRing` only ever in the ring, so neither can stand in for
  // the other.
  auto *spilled = array(1);
  auto *inRing = array(1);
  whiten();
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  for (uint32_t i = 0; i < kRingCapacity; ++i)
    mutator.push(spilled, gc.cycleSeq());
  // Fills the ring, so this push spills all of `spilled` and then lands alone.
  mutator.push(inRing, gc.cycleSeq());
  ASSERT_EQ(kRingCapacity, gc.barrierWorklistSlow().size());

  runToCertification();
  EXPECT_TRUE(SharedGC::isMarked(spilled)) << "the worklist half";
  EXPECT_TRUE(SharedGC::isMarked(inRing)) << "the ring half";
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow());
}

TEST_F(SharedCertifyTest, Certify_TracesWhatARingEntryTransitivelyReaches) {
  // A cell recovered from a ring is enqueued, not just marked, so a later
  // round traces its children.
  auto *head = array(1);
  auto *tail = array(1);
  auto *unreached = array(1);
  link(head, 0, tail);
  whiten();
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  mutator.push(head, gc.cycleSeq());
  EXPECT_GT(runToCertification(), 1u) << "the first round must fail";
  EXPECT_TRUE(SharedGC::isMarked(head));
  EXPECT_TRUE(SharedGC::isMarked(tail));
  EXPECT_FALSE(SharedGC::isMarked(unreached));
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow());
}

TEST_F(SharedCertifyTest, Cycle_EndsWhileMutatorsAreStillPushing) {
  // Producers spill continuously while the collector opens, certifies, closes
  // and sweeps 300 cycles. Without turning marking off inside certifyRound's
  // critical section, a spill lands after the last drain and endCycle trips.
  //
  // Checked in every build: each cycle turns marking off, nothing lands in the
  // barrier worklist after certification, and the live count is exact.
  auto *root = array(1);
  auto *child = array(1);
  link(root, 0, child);
  whiten();

  std::atomic<bool> done{false};
  std::atomic<uint32_t> pushed{0};
  auto produce = [&] {
    while (!done.load(std::memory_order_acquire)) {
      // Attaching and detaching in the loop puts registration and the
      // unregister flush on top of certification.
      SharedMutator mutator{gc, runtime};
      for (uint32_t i = 0; i < kRingCapacity; ++i) {
        // Check whether marking is on exactly as the barrier will: one load
        // gives both the answer and the stamp, and an even one means do
        // nothing.
        const uint32_t seq = gc.cycleSeq();
        if (seq % 2 == 0)
          continue;
        mutator.push(root, seq);
        mutator.push(child, seq);
        pushed.fetch_add(2, std::memory_order_relaxed);
      }
    }
  };
  std::thread first(produce);
  std::thread second(produce);

  // At least 300 cycles, and on until the producers have pushed during one: a
  // thread just started need not run before a quick loop ends. No ASSERT here,
  // which would return with the producers still running.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(60);
  for (uint32_t i = 0; i < 300 ||
       (pushed.load(std::memory_order_relaxed) == 0 &&
        std::chrono::steady_clock::now() < deadline);
       ++i) {
    gc.beginCycle();
    gc.markRoots({root});
    runToCertification();
    if (gc.markingOn()) {
      ADD_FAILURE() << "cycle " << i << " left marking on";
      gc.endCycle();
      break;
    }
    // With marking off, a spill drops what it drains, so the worklist stays
    // empty until endCycle, whose own check of this is an assert.
    if (!gc.barrierWorklistSlow().empty()) {
      ADD_FAILURE() << "cycle " << i << ": a spill landed after certification";
      gc.drainBarrierWorklist();
      gc.drainMarkWorklist();
    }
    gc.endCycle();
    gc.sweep();
    EXPECT_EQ(2u, gc.liveCellCountSlow()) << "at cycle " << i;
  }
  done.store(true, std::memory_order_release);
  first.join();
  second.join();

  EXPECT_GT(pushed.load(), 0u) << "the producers never saw marking on";
  EXPECT_EQ(0u, gc.mutatorCountSlow());
}

TEST_F(SharedCertifyTest, Concurrent_ChurnDoesNotStopCertification) {
  // Termination under load. A producer pushes already-marked cells as fast as
  // it can; with an emptiness measure the loop below never exits.
  auto *root = array(1);
  whiten();
  std::atomic<bool> done{false};
  std::atomic<uint32_t> pushed{0};

  gc.beginCycle();
  gc.markRoots({root});
  gc.drainMarkWorklist();
  std::thread churn([&] {
    SharedMutator mutator{gc, runtime};
    while (!done.load(std::memory_order_acquire)) {
      const uint32_t seq = gc.cycleSeq();
      if (seq % 2 == 1) {
        mutator.push(root, seq);
        pushed.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });
  // Without this the collector certifies before the thread has even
  // constructed its SharedMutator, and the test exercises no churn at all.
  while (pushed.load(std::memory_order_relaxed) == 0) {
  }

  const uint32_t rounds = runToCertification();
  done.store(true, std::memory_order_release);
  churn.join();

  EXPECT_GT(pushed.load(), 0u);
  EXPECT_GE(rounds, 1u);
  EXPECT_FALSE(gc.markingOn());
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(1u, gc.liveCellCountSlow());
}

} // namespace
