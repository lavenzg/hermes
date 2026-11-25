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

/// Free everything: one complete cycle with no roots marks nothing, so its
/// sweep frees the heap. A legitimate collection rather than a teardown
/// special case -- beginCycle's reset clears whatever an abandoned cycle left
/// behind, which a bare sweep would misread as garbage.
static void freeAll(SharedGC &gc) {
  // A test that failed midway can leave its cycle open.
  if (gc.markingOn())
    gc.abortCycle();
  gc.beginCycle();
  while (!gc.certifyRound())
    gc.drainMarkWorklist();
  gc.endCycle();
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
    from->initField(i, SharedValue::encodePointer(to));
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

TEST_F(SharedCertifyTest, BeginCycle_ResetsMarksLeftByAllocation) {
  // The reason beginCycle resets at all. Cells are born marked whether or not
  // a cycle is running, and only a sweep clears a mark, so a cell allocated
  // between cycles reaches the next beginCycle already marked. A mark tells
  // accept() the cell is already enqueued, so its slots are never scanned --
  // and anything reachable only through it is swept while still reachable.
  //
  // Every step here is a legitimate driver operation: no bare sweep, and each
  // cell is reachable from a root at the time of each collection.
  auto *child = array(0);
  gc.beginCycle();
  gc.markRoots({child});
  runToCertification();
  gc.endCycle();
  gc.sweep();
  ASSERT_EQ(1u, gc.liveCellCountSlow()) << "child survived cycle 1";
  ASSERT_FALSE(SharedGC::isMarked(child)) << "survivors end a sweep white";

  // Ordinary work between cycles: allocate, and point the new cell at an old
  // one. No barrier runs, because marking is off.
  auto *root = array(1);
  ASSERT_TRUE(SharedGC::isMarked(root)) << "born marked";
  link(root, 0, child);

  gc.beginCycle();
  EXPECT_FALSE(SharedGC::isMarked(root)) << "the bulk reset whitened it";
  gc.markRoots({root});
  runToCertification();
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow())
      << "child is reachable through root and must survive";
  EXPECT_EQ(child, root->at(0).getPointer());
}

TEST_F(SharedCertifyTest, BeginCycle_ResetsOversizedRegionsToo) {
  // Cells over the largest size bucket get their own regions on a separate
  // list, and the reset is the only thing that walks it. Every other test
  // allocates small, so deleting that walk passes the whole suite.
  auto *big = array(2000); // allocationSize is well over kMaxBucketSize
  auto *child = array(0);
  link(big, 0, child);

  // Deliberately no whiten(): big's birth mark is what the reset must clear.
  gc.beginCycle();
  EXPECT_FALSE(SharedGC::isMarked(big)) << "an oversized region was not reset";
  gc.markRoots({big});
  runToCertification();
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow()) << "child was reachable through big";
}

TEST_F(SharedCertifyTest, Allocation_AfterCertificationSurvivesTheSweep) {
  // The window that makes unconditional black allocation mandatory rather than
  // merely harmless. A cell born after marking is turned off is marked by
  // nothing -- no trace will reach it and no barrier runs -- so were it born
  // white the very next sweep would free it.
  auto *root = array(0);
  gc.beginCycle();
  gc.markRoots({root});
  runToCertification();
  gc.endCycle();

  auto *newborn = array(0);
  EXPECT_TRUE(SharedGC::isMarked(newborn)) << "must be born marked";
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow()) << "the sweep freed a newborn";
}

TEST_F(SharedCertifyTest, Abort_TurnsMarkingOffForAnUncertifiedCycle) {
  auto *survivor = array(0);
  whiten();
  gc.beginCycle();
  ASSERT_TRUE(gc.markingOn());
  gc.abortCycle();
  EXPECT_FALSE(gc.markingOn())
      << "an abandoned cycle must still turn marking off";

  // And the heap is usable again immediately, with something in it to lose.
  gc.beginCycle();
  gc.markRoots({survivor});
  runToCertification();
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(1u, gc.liveCellCountSlow()) << "the survivor did not survive";
}

#ifdef NDEBUG
TEST_F(SharedCertifyTest, Sweep_AfterAnUncertifiedCycle_FreesNothing) {
  // Debug builds assert here. A release build must still not act on a partial
  // trace, after abortCycle or after endCycle on a cycle that never certified.
  auto *root = array(1);
  auto *child = array(1);
  link(root, 0, child);
  whiten();

  gc.beginCycle();
  gc.markRoots({root});
  gc.abortCycle();
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow()) << "swept after abortCycle";

  gc.beginCycle();
  gc.markRoots({root});
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow()) << "swept after an uncertified cycle";
}
#endif

TEST_F(SharedCertifyTest, Abort_DiscardsWorklistsEndCycleWouldDemand) {
  // endCycle insists both worklists are drained, because anything left in them
  // is a live cell about to be swept. Abandoning has no such obligation: the
  // records belong to a cycle that will never certify, and nothing is swept on
  // the strength of them.
  auto *cell = array(1);
  whiten();
  SharedMutator mutator{gc, runtime};

  gc.beginCycle();
  mutator.push(cell, gc.cycleSeq());
  ASSERT_EQ(1u, gc.stealBarrierEntries());
  gc.markRoots({cell});
  ASSERT_FALSE(gc.barrierWorklistSlow().empty());

  gc.abortCycle();
  EXPECT_FALSE(gc.markingOn());

  // The discard itself: the next cycle starts with nothing carried over. An
  // abortCycle that kept the worklists would pass on turning marking off alone.
  gc.beginCycle();
  EXPECT_TRUE(gc.barrierWorklistSlow().empty()) << "the worklist carried over";
  EXPECT_EQ(0u, gc.stealBarrierEntries()) << "a stale ring entry was accepted";
  runToCertification();
  gc.endCycle();
  gc.sweep();
}

TEST_F(SharedCertifyTest, Abort_MidTrace_DoesNotLoseTheUntracedSubgraph) {
  // Why a cycle has to start from an all-white heap. A mark bit means both
  // "reachable" and
  // "already enqueued, so do not trace its children", and the second is only
  // true within one cycle. Leaving a half-traced cycle's marks behind makes
  // the next cycle's accept() refuse to enqueue `root`, so `child` is never
  // traced and the sweep frees it while it is still reachable.
  //
  // The edge from a marked cell to a white one is the whole point: without it
  // this passes against the broken version.
  auto *root = array(1);
  auto *child = array(0);
  link(root, 0, child);
  whiten();

  gc.beginCycle();
  gc.markRoots({root}); // marked and enqueued, not yet traced
  gc.abortCycle();
  EXPECT_TRUE(SharedGC::isMarked(root)) << "the partial trace's mark survives";
  EXPECT_FALSE(SharedGC::isMarked(child));

  // beginCycle's bulk reset is what clears it. Without that reset, accept()
  // would refuse to enqueue root here and child would never be traced.
  gc.beginCycle();
  EXPECT_FALSE(SharedGC::isMarked(root)) << "the reset whitened it";
  gc.markRoots({root});
  runToCertification();
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow()) << "the untraced subgraph was lost";
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
      gc.abortCycle();
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
