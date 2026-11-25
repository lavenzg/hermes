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

/// Wait up to ten seconds for \p progress to leave zero: a thread just
/// started need not run before a quick loop of cycles ends.
void waitForProgress(const std::atomic<uint32_t> &progress) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (progress.load(std::memory_order_relaxed) == 0 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
}

struct SharedWriteBarrierTest : public RuntimeTestFixture {
  SharedGC gc{kAmpleBudget};
  SharedMutator mutator{gc, runtime};

  ~SharedWriteBarrierTest() override {
    // A rootless cycle, not a bare sweep: see the other suites' freeAll.
    gc.beginCycle();
    while (!gc.certifyRound())
      gc.drainMarkWorklist();
    gc.endCycle();
    gc.sweep();
  }

  SharedArray *array(uint32_t length) {
    SharedArray *arr = SharedArray::create(gc, length);
    EXPECT_NE(nullptr, arr);
    return arr;
  }

  /// Clear the birth marks, so a cycle starts from an all-white heap.
  void whiten() {
    const size_t live = gc.liveCellCountSlow();
    gc.sweep();
    ASSERT_EQ(live, gc.liveCellCountSlow());
  }

  uint32_t runToCertification() {
    uint32_t rounds = 0;
    do {
      gc.drainMarkWorklist();
      ++rounds;
    } while (!gc.certifyRound());
    return rounds;
  }
};

TEST_F(SharedWriteBarrierTest, Store_WithMarkingOff_RecordsNothing) {
  // The common path: nothing reaches the ring.
  auto *holder = array(1);
  auto *victim = array(0);
  holder->initField(0, SharedValue::encodePointer(victim));
  whiten();

  holder->set(0, SharedValue::empty(), mutator);
  EXPECT_EQ(0u, mutator.ringSizeSlow());
}

TEST_F(SharedWriteBarrierTest, Store_WithMarkingOn_RecordsTheOverwrittenCell) {
  auto *holder = array(1);
  auto *victim = array(0);
  holder->initField(0, SharedValue::encodePointer(victim));
  whiten();

  gc.beginCycle();
  holder->set(0, SharedValue::empty(), mutator);
  EXPECT_EQ(1u, mutator.ringSizeSlow());

  EXPECT_GT(runToCertification(), 1u) << "the record must fail a round first";
  gc.endCycle();
  gc.sweep();
}

TEST_F(SharedWriteBarrierTest, Store_OverAnEmptySlot_RecordsNothing) {
  // There is no overwritten reference, so nothing is pushed.
  auto *holder = array(1);
  auto *other = array(0);
  whiten();

  gc.beginCycle();
  holder->set(0, SharedValue::encodePointer(other), mutator);
  EXPECT_EQ(0u, mutator.ringSizeSlow());

  runToCertification();
  gc.endCycle();
  gc.sweep();
}

TEST_F(SharedWriteBarrierTest, Store_TouchesNoMarkBit) {
  // The mutator records; only the collector marks. Black allocation is the one
  // mark bit a mutator writes, and it belongs to a different cell.
  auto *holder = array(1);
  auto *victim = array(0);
  holder->initField(0, SharedValue::encodePointer(victim));
  whiten();

  gc.beginCycle();
  holder->set(0, SharedValue::empty(), mutator);
  EXPECT_FALSE(SharedGC::isMarked(victim)) << "the barrier marked, it must not";

  runToCertification();
  EXPECT_TRUE(SharedGC::isMarked(victim)) << "certification should have";
  gc.endCycle();
  gc.sweep();
}

TEST_F(SharedWriteBarrierTest, Store_RetainsACellMovedIntoABlackCell) {
  // The case the barrier exists for, and the one this suite previously
  // asserted *broken*.
  //
  // `hidden` is allocated during the cycle, so it is born marked and will
  // never be traced. Moving `treasure` into it and then clearing the only
  // white reference leaves `treasure` reachable from nothing the collector
  // will look at. Only the record of the overwritten slot saves it.
  auto *visible = array(1);
  auto *treasure = array(0);
  visible->initField(0, SharedValue::encodePointer(treasure));
  whiten();

  gc.beginCycle();
  auto *hidden = array(1); // born marked, therefore never traced
  // initField, not set: `hidden` is not published until markRoots below, and
  // using set() here would push `treasure` itself, hiding whether the one
  // store that matters records the old value or the new one.
  hidden->initField(0, SharedValue::encodePointer(treasure));
  visible->set(0, SharedValue::empty(), mutator);

  gc.markRoots({visible, hidden});
  EXPECT_GT(runToCertification(), 1u) << "the record must fail a round first";
  gc.endCycle();
  gc.sweep();

  EXPECT_EQ(3u, gc.liveCellCountSlow()) << "treasure was swept";
  EXPECT_EQ(treasure, hidden->at(0).getPointer());
}

TEST_F(SharedWriteBarrierTest, Store_AfterTheCycleEnded_IsDroppedNotAccepted) {
  // A barrier reads the counter, is preempted for the rest of the cycle, and
  // completes its push afterwards. The stamp is what keeps that entry from
  // being handed to a tracer, since by then the cell may be swept.
  auto *cell = array(1);
  whiten();

  gc.beginCycle();
  const uint32_t stale = gc.cycleSeq();
  runToCertification();
  gc.endCycle();
  gc.sweep();

  // The push finally lands, stamped with the cycle that has already gone.
  mutator.push(cell, stale);

  gc.beginCycle();
  EXPECT_EQ(0u, gc.stealBarrierEntries()) << "a stale entry was accepted";
  runToCertification();
  gc.endCycle();
  gc.sweep();
}

TEST_F(
    SharedWriteBarrierTest,
    Concurrent_TraceMissesTheMoveAndTheBarrierSaves) {
  // The interleaving the barrier exists for, driven across two threads rather
  // than hoped for: the collector traces `dst` while it is empty, the producer
  // then moves `treasure` out of `src` into `dst`, and the collector traces
  // `src` and finds nothing. The clearing store's record is the only thing
  // that names `treasure` anywhere the collector will look.
  auto *src = array(1);
  auto *dst = array(1);
  auto *treasure = array(0);
  src->initField(0, SharedValue::encodePointer(treasure));
  whiten();

  std::atomic<bool> done{false}, go{false}, moved{false};
  std::thread producer([&] {
    SharedMutator local{gc, runtime};
    while (!done.load(std::memory_order_acquire)) {
      if (!go.load(std::memory_order_acquire))
        continue;
      const SharedValue held = src->at(0);
      dst->set(0, held, local);
      src->set(0, SharedValue::empty(), local);
      moved.store(true, std::memory_order_release);
      go.store(false, std::memory_order_relaxed);
    }
  });

  gc.beginCycle();
  gc.markRoots({dst});
  gc.drainMarkWorklist(); // dst is traced now, while it is still empty
  go.store(true, std::memory_order_release);
  while (!moved.load(std::memory_order_acquire)) {
  }
  gc.markRoots({src}); // and src is traced after the move, also empty
  const uint32_t rounds = runToCertification();
  done.store(true, std::memory_order_release);
  producer.join();

  EXPECT_GT(rounds, 1u) << "nothing had to be recovered, so the trace found "
                           "treasure and the interleaving was not reached";
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(3u, gc.liveCellCountSlow()) << "treasure was swept";
}

TEST_F(SharedWriteBarrierTest, Barrier_RecordedButUnmarked_FailsTheRound) {
  // The push landed and the store landed, so the trace sees an empty slot.
  // The round must not certify while the recorded cell is still unmarked.
  auto *holder = array(1);
  auto *victim = array(0);
  holder->initField(0, SharedValue::encodePointer(victim));
  whiten();

  gc.beginCycle();
  holder->set(0, SharedValue::empty(), mutator);
  gc.markRoots({holder});
  gc.drainMarkWorklist();
  EXPECT_FALSE(SharedGC::isMarked(victim)) << "the trace cannot have found it";
  EXPECT_FALSE(gc.certifyRound()) << "an outstanding record must fail a round";

  runToCertification();
  gc.endCycle();
  gc.sweep();
  EXPECT_EQ(2u, gc.liveCellCountSlow());
}

TEST_F(SharedWriteBarrierTest, Barrier_SpillsWhenItOverrunsTheRing) {
  // More barriered stores in one cycle than a ring holds, so the barrier drives
  // the spill itself -- which takes barrierMtx_ between the pre-read and the
  // store, the one place a mutator waits on the collector. Nothing else in the
  // suite reaches that path through a slot store.
  //
  // Every overwritten cell is retained, which is what a snapshot barrier
  // promises: conservative, not precise.
  constexpr uint32_t kCount = SharedBarrierRing::kCapacity * 2 + 3;
  auto *holder = array(kCount);
  for (uint32_t i = 0; i < kCount; ++i)
    holder->initField(i, SharedValue::encodePointer(array(0)));
  whiten();

  gc.beginCycle();
  for (uint32_t i = 0; i < kCount; ++i)
    holder->set(i, SharedValue::empty(), mutator);
  gc.markRoots({holder});
  EXPECT_GT(runToCertification(), 1u);
  gc.endCycle();
  gc.sweep();

  EXPECT_EQ(kCount + 1, gc.liveCellCountSlow()) << "spilled records were lost";
}

TEST_F(SharedWriteBarrierTest, Concurrent_TwoRuntimesStoringOneSlot) {
  // Two runtimes writing the same slot, so each writer's pre-read can return
  // the other's value. Both candidate values stay rooted, so a lost record
  // cannot show here: the test is for TSan, which checks that each seq_cst
  // pre-read is ordered after the store it reads.
  auto *anchor = array(2);
  auto *shared = array(1);
  auto *red = array(0);
  auto *blue = array(0);
  anchor->initField(0, SharedValue::encodePointer(red));
  anchor->initField(1, SharedValue::encodePointer(blue));
  shared->initField(0, SharedValue::encodePointer(red));
  whiten();

  std::atomic<bool> done{false};
  std::atomic<uint32_t> stores{0};
  auto write = [&](SharedGCCell *value) {
    SharedMutator local{gc, runtime};
    while (!done.load(std::memory_order_acquire)) {
      shared->set(0, SharedValue::encodePointer(value), local);
      stores.fetch_add(1, std::memory_order_relaxed);
    }
  };
  std::thread first(write, red);
  std::thread second(write, blue);
  waitForProgress(stores);

  size_t survivors = 4;
  for (uint32_t i = 0; i < 200 && survivors == 4; ++i) {
    gc.beginCycle();
    while (!gc.pollInFlightBarriers())
      std::this_thread::yield();
    gc.beginSecondInFlightPass();
    while (!gc.pollInFlightBarriers())
      std::this_thread::yield();
    gc.markRoots({anchor, shared});
    runToCertification();
    gc.endCycle();
    gc.sweep();
    survivors = gc.liveCellCountSlow();
  }
  done.store(true, std::memory_order_release);
  first.join();
  second.join();

  EXPECT_EQ(4u, survivors);
  EXPECT_GT(stores.load(), 0u);
}

TEST_F(SharedWriteBarrierTest, Concurrent_MoveBetweenRootsRetainsTheTarget) {
  // A producer shuttles the only reference to `treasure` between two rooted
  // slots while the collector runs whole cycles. `treasure` is reachable from
  // one of them at every instant -- the producer writes the destination before
  // clearing the source -- so it must never be swept.
  //
  // The interleaving that needs the barrier: the collector traces `dst` while
  // it is empty, the producer then moves `treasure` into `dst` and clears
  // `src`, and the collector traces `src` and finds nothing. Only the record
  // of the cleared slot saves it. If the barrier is wrong the live count drops
  // and the collector goes on to trace freed memory, which ASan reports.
  //
  // markRoots requires the wait after beginCycle: the producer can be inside
  // set() when a cycle begins. On x86 and ARMv8 this test would keep
  // `treasure` without the wait, since the clearing store tests the flag only
  // after the destination store lands, but the C++ memory model does not order
  // that store before the tracer's read without it. Either way, the test is
  // not evidence for the wait.
  //
  // Both slots are roots on purpose. Modelling the producer's own reference as
  // invisible to the collector would make this a test of a missing root set,
  // not of the barrier.
  auto *src = array(1);
  auto *dst = array(1);
  auto *treasure = array(0);
  src->initField(0, SharedValue::encodePointer(treasure));
  whiten();

  std::atomic<bool> done{false};
  std::atomic<uint32_t> moves{0};
  std::thread producer([&] {
    SharedMutator local{gc, runtime};
    while (!done.load(std::memory_order_acquire)) {
      SharedArray *from = src, *to = dst;
      if (from->at(0).isEmpty())
        std::swap(from, to);
      const SharedValue held = from->at(0);
      if (held.isEmpty())
        continue;
      // Destination first, so the reference is never in neither slot.
      to->set(0, held, local);
      from->set(0, SharedValue::empty(), local);
      moves.fetch_add(1, std::memory_order_relaxed);
    }
  });
  waitForProgress(moves);

  size_t survivors = 3;
  uint32_t failedAt = 0;
  uint32_t roundsSavedBy = 0;
  for (uint32_t i = 0; i < 300 && survivors == 3; ++i) {
    gc.beginCycle();
    while (!gc.pollInFlightBarriers())
      std::this_thread::yield();
    gc.beginSecondInFlightPass();
    while (!gc.pollInFlightBarriers())
      std::this_thread::yield();
    gc.markRoots({src, dst});
    // A round fails only when certification marked something the trace did not
    // reach, so this counts the times the barrier was actually load-bearing.
    // Without it, "300 cycles passed" is compatible with never hitting the
    // window at all.
    roundsSavedBy += runToCertification() - 1;
    gc.endCycle();
    gc.sweep();
    survivors = gc.liveCellCountSlow();
    if (survivors != 3)
      failedAt = i;
  }
  // Stop and join before asserting: an ASSERT_ here would return with the
  // producer still running and std::terminate on the unjoined thread.
  done.store(true, std::memory_order_release);
  producer.join();

  EXPECT_EQ(3u, survivors) << "a live cell was swept at cycle " << failedAt;
  EXPECT_GT(moves.load(), 0u);
  // Deliberately not asserted: whether the window is hit is up to the
  // scheduler, and requiring it here made the test flake. The deterministic
  // version above is what proves the barrier is load-bearing; this one covers
  // the schedule space around it.
  if (roundsSavedBy == 0)
    GTEST_LOG_(INFO) << "the window was not reached in this run";
}

} // namespace
