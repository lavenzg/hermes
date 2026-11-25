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
#include <cstdint>
#include <memory>
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

struct SharedMutatorTest : public RuntimeTestFixture {
  SharedGC gc{kAmpleBudget};

  ~SharedMutatorTest() override {
    freeAll(gc);
  }

  SharedArray *array(uint32_t length) {
    SharedArray *arr = SharedArray::create(gc, length);
    EXPECT_NE(nullptr, arr);
    return arr;
  }
};

TEST_F(SharedMutatorTest, CycleSeq_IsOddOnlyDuringACycle) {
  EXPECT_EQ(0u, gc.cycleSeq() % 2);
  gc.beginCycle();
  const uint32_t first = gc.cycleSeq();
  EXPECT_EQ(1u, first % 2);
  gc.drainBarrierWorklist();
  gc.endCycle();
  EXPECT_EQ(0u, gc.cycleSeq() % 2);

  gc.beginCycle();
  EXPECT_NE(first, gc.cycleSeq()) << "each cycle needs its own identity";
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Runtime_IsTheOneItWasConstructedWith) {
  // The collector reaches a runtime's box map -- its root set -- only through
  // this reference, so a mutator naming the wrong runtime would have the
  // collector scan the wrong map and sweep cells the right one still holds.
  SharedMutator mutator{gc, runtime};
  EXPECT_EQ(&runtime, &mutator.runtime());
}

TEST_F(SharedMutatorTest, Register_TracksEveryLiveRuntime) {
  EXPECT_EQ(0u, gc.mutatorCountSlow());
  {
    SharedMutator a{gc, runtime};
    EXPECT_EQ(1u, gc.mutatorCountSlow());
    {
      SharedMutator b{gc, runtime};
      EXPECT_EQ(2u, gc.mutatorCountSlow());
    }
    EXPECT_EQ(1u, gc.mutatorCountSlow());
  }
  EXPECT_EQ(0u, gc.mutatorCountSlow());
}

TEST_F(SharedMutatorTest, Unregister_LeavesTheOthersReachable) {
  // Deregistration swaps with the back, so it must not lose an unrelated
  // runtime's registration.
  auto *cell = array(1);
  auto first = std::make_unique<SharedMutator>(gc, runtime);
  SharedMutator second{gc, runtime};
  SharedMutator third{gc, runtime};
  first.reset();
  ASSERT_EQ(2u, gc.mutatorCountSlow());

  gc.beginCycle();
  second.push(cell, gc.cycleSeq());
  third.push(cell, gc.cycleSeq());
  EXPECT_EQ(2u, gc.stealBarrierEntries());
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Steal_CollectsFromEveryRuntime) {
  auto *a = array(1);
  auto *b = array(1);
  SharedMutator one{gc, runtime};
  SharedMutator two{gc, runtime};

  gc.beginCycle();
  one.push(a, gc.cycleSeq());
  two.push(b, gc.cycleSeq());
  ASSERT_EQ(2u, gc.stealBarrierEntries());

  const std::vector<SharedGCCell *> stolen = gc.barrierWorklistSlow();
  ASSERT_EQ(2u, stolen.size());
  EXPECT_TRUE(
      (stolen[0] == a && stolen[1] == b) || (stolen[0] == b && stolen[1] == a));
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Steal_LeavesTheRingsEmpty) {
  auto *cell = array(1);
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  mutator.push(cell, gc.cycleSeq());
  ASSERT_EQ(1u, gc.stealBarrierEntries());
  EXPECT_EQ(0u, gc.stealBarrierEntries());
  EXPECT_EQ(1u, gc.barrierWorklistSlow().size());
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Steal_DropsEntriesFromAnotherCycle) {
  // The stamp is the only thing standing between a cell pushed in an earlier
  // cycle, since swept, and the tracer dereferencing it.
  auto *cell = array(1);
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  const uint32_t seq = gc.cycleSeq();
  mutator.push(cell, seq - 2);
  mutator.push(cell, seq);
  mutator.push(cell, seq - 4);
  EXPECT_EQ(1u, gc.stealBarrierEntries());
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Steal_AccumulatesAcrossRounds) {
  // Certification steals repeatedly within one cycle, and each round appends
  // rather than replacing.
  auto *cell = array(1);
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  mutator.push(cell, gc.cycleSeq());
  ASSERT_EQ(1u, gc.stealBarrierEntries());
  mutator.push(cell, gc.cycleSeq());
  ASSERT_EQ(1u, gc.stealBarrierEntries());
  EXPECT_EQ(2u, gc.barrierWorklistSlow().size());
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Push_SpillsWhenTheRingFills) {
  // The overflow path: the ring holds kCapacity, and push number kCapacity + 1
  // has to drain it into the barrier worklist rather than overwrite.
  auto *cell = array(1);
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  for (uint32_t i = 0; i < kRingCapacity; ++i)
    mutator.push(cell, gc.cycleSeq());
  EXPECT_TRUE(gc.barrierWorklistSlow().empty()) << "spilled too early";

  mutator.push(cell, gc.cycleSeq());
  EXPECT_EQ(kRingCapacity, gc.barrierWorklistSlow().size());

  // The one that triggered the spill is still in the ring.
  EXPECT_EQ(1u, gc.stealBarrierEntries());
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Push_KeepsSpillingPastTheFirstSpill) {
  // Several capacities' worth, so a ring that spilled once but never recovered
  // its cursor is caught.
  constexpr uint32_t kTotal = kRingCapacity * 4 + 5;
  auto *cell = array(1);
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  for (uint32_t i = 0; i < kTotal; ++i)
    mutator.push(cell, gc.cycleSeq());
  const size_t spilled = gc.barrierWorklistSlow().size();
  const size_t remaining = gc.stealBarrierEntries();

  // A full ring every kRingCapacity pushes, and what is left after the last.
  EXPECT_EQ(kRingCapacity * 4, spilled);
  EXPECT_EQ(5u, remaining);
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Unregister_FlushesTheRingMidCycle) {
  // A Runtime torn down mid-cycle must not take its SATB entries with it: a
  // cell whose only surviving reference was copied into another runtime's
  // already-traced slot would otherwise go unmarked, and sweep would free it.
  auto *a = array(1);
  auto *b = array(1);
  SharedMutator survivor{gc, runtime};

  gc.beginCycle();
  {
    SharedMutator dying{gc, runtime};
    // More than a ring's worth, so both what it spilled and what it still held
    // have to survive it.
    for (uint32_t i = 0; i < kRingCapacity; ++i)
      dying.push(a, gc.cycleSeq());
    dying.push(b, gc.cycleSeq());
    ASSERT_EQ(kRingCapacity, gc.barrierWorklistSlow().size());
  }
  ASSERT_EQ(1u, gc.mutatorCountSlow());

  const std::vector<SharedGCCell *> flushed = gc.barrierWorklistSlow();
  ASSERT_EQ(kRingCapacity + 1, flushed.size()) << "the ring was dropped";
  EXPECT_EQ(b, flushed.back());
  EXPECT_EQ(0u, gc.stealBarrierEntries()) << "nothing left to steal";
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Unregister_WithNoCycleRunning) {
  // Runtimes are created and destroyed with no collection in flight, which is
  // the case the flush above must not assume away.
  auto *cell = array(1);
  {
    SharedMutator mutator{gc, runtime};
    mutator.push(cell, gc.cycleSeq());
  }
  EXPECT_EQ(0u, gc.mutatorCountSlow());
}

TEST_F(SharedMutatorTest, Spill_WithNoCycleDiscardsTheRing) {
  // A barrier preempted across the end of a cycle arrives here with no
  // MarkState. Its cells may already be swept and their segments unmapped, so
  // the ring has to be consumed without being read -- and without reaching for
  // a mutex that no longer exists, which is why barrierMtx_ is not in
  // MarkState.
  auto *cell = array(1);
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  const uint32_t stale = gc.cycleSeq();
  gc.drainBarrierWorklist();
  gc.endCycle();

  for (uint32_t i = 0; i < kRingCapacity + 1; ++i)
    mutator.push(cell, stale);
  // The spill discarded rather than kept: only the push that followed it is
  // still there. Asserting the steal alone would pass on the stamp filter even
  // if the ring had never been drained.
  EXPECT_EQ(1u, mutator.ringSizeSlow());

  gc.beginCycle();
  EXPECT_EQ(0u, gc.stealBarrierEntries()) << "a stale entry was accepted";
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Unregister_FlushesOnlyTheCurrentCycleStamps) {
  // The flush is a drain, so it consumes the whole ring -- but it must keep
  // only this cycle's entries. An older one may name a cell already swept.
  auto *live = array(1);
  auto *stale = array(1);
  SharedMutator survivor{gc, runtime};

  gc.beginCycle();
  {
    SharedMutator dying{gc, runtime};
    dying.push(stale, gc.cycleSeq() - 2);
    dying.push(live, gc.cycleSeq());
  }
  const std::vector<SharedGCCell *> flushed = gc.barrierWorklistSlow();
  ASSERT_EQ(1u, flushed.size());
  EXPECT_EQ(live, flushed[0]);
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Register_MidCycleRuntimeIsStolenFrom) {
  // A Runtime can attach after a cycle has turned marking on. It is stamped
  // with the same counter, so the very next steal has to reach it.
  auto *cell = array(1);
  gc.beginCycle();
  SharedMutator latecomer{gc, runtime};
  latecomer.push(cell, gc.cycleSeq());
  EXPECT_EQ(1u, gc.stealBarrierEntries());
  gc.drainBarrierWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, DrainBarrierWorklist_ReportsWhatItNewlyMarked) {
  // The measure certification runs on. An emptiness test would be weaker:
  // churn over already-marked cells never empties anything.
  auto *a = array(1);
  auto *b = array(1);
  SharedMutator mutator{gc, runtime};
  gc.beginCycle();
  gc.endCycle();
  gc.sweep(); // clears the birth marks

  gc.beginCycle();
  mutator.push(a, gc.cycleSeq());
  mutator.push(b, gc.cycleSeq());
  mutator.push(a, gc.cycleSeq());
  ASSERT_EQ(3u, gc.stealBarrierEntries());
  EXPECT_EQ(2u, gc.drainBarrierWorklist()) << "a counted once, not twice";
  EXPECT_TRUE(SharedGC::isMarked(a));
  EXPECT_TRUE(SharedGC::isMarked(b));

  mutator.push(a, gc.cycleSeq());
  ASSERT_EQ(1u, gc.stealBarrierEntries());
  EXPECT_EQ(0u, gc.drainBarrierWorklist()) << "nothing new";
  gc.drainMarkWorklist();
  gc.endCycle();
}

TEST_F(SharedMutatorTest, Concurrent_SpillStealAndAttachInOneCycle) {
  // Two runtimes push and spill continuously while the collector steals and
  // drains, and each attaches and detaches mid-cycle, so unregisterMutator's
  // flush races the steal. Schedule-independent postcondition: it completes,
  // clean under ASan, with the registry empty and the heap intact.
  //
  // The cycle deliberately does not *end* while they run. It cannot: a spill
  // landing between the collector's last drain and endCycle leaves the barrier
  // worklist non-empty, which is precisely the two-step-check unsoundness that
  // certifyRound exists to close by consuming and turning marking off in one
  // critical section. That test belongs with certifyRound.
  //
  // ASan cannot see a missing lock directly, so under ASan this is only a
  // probabilistic guard; TSan sees one directly. It is contended on purpose.
  auto *cell = array(1);
  std::atomic<bool> done{false};
  std::atomic<uint32_t> pushed{0};

  gc.beginCycle();
  auto produce = [&] {
    while (!done.load(std::memory_order_acquire)) {
      // Attaching and detaching inside the loop is the point: it puts
      // registration, the flush, and the steal on top of each other.
      SharedMutator mutator{gc, runtime};
      for (uint32_t i = 0; i < kRingCapacity * 2; ++i) {
        mutator.push(cell, gc.cycleSeq());
        pushed.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };
  std::thread first(produce);
  std::thread second(produce);

  // Run until the producers have pushed enough to have spilled several times,
  // not for a fixed count alone: that can finish before either thread is
  // scheduled, leaving nothing to race and failing the check below.
  for (uint32_t i = 0;
       i < 2000 || pushed.load(std::memory_order_relaxed) < kRingCapacity * 4;
       ++i) {
    gc.stealBarrierEntries();
    gc.drainBarrierWorklist();
    gc.drainMarkWorklist();
  }
  done.store(true, std::memory_order_release);
  first.join();
  second.join();

  EXPECT_EQ(0u, gc.mutatorCountSlow()) << "both producers deregistered";
  gc.stealBarrierEntries();
  gc.drainBarrierWorklist();
  gc.drainMarkWorklist();
  EXPECT_GT(pushed.load(), 0u);
  gc.endCycle();
}

} // namespace
