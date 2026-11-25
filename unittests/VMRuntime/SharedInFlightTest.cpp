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

namespace {

using namespace hermes;
using namespace hermes::vm;

constexpr size_t kAmpleBudget = 8 << 20;
constexpr uint32_t kRingCapacity = SharedBarrierRing::kCapacity;

/// Most tests here make each barrier step its own call, so a cycle can begin
/// between a window's test of the marking flag and the store it guards -- the
/// case the in-flight check exists for -- without a second thread.
struct SharedInFlightTest : public RuntimeTestFixture {
  SharedGC gc{kAmpleBudget};

  ~SharedInFlightTest() override {
    // A test that failed midway can leave its cycle open.
    if (gc.markingOn())
      gc.abortCycle();
    // A rootless cycle frees whatever a test allocated.
    gc.beginCycle();
    while (!gc.certifyRound())
      gc.drainMarkWorklist();
    gc.endCycle();
    gc.sweep();
  }
};

TEST_F(SharedInFlightTest, Wait_IsOverAtOnceWithNoRuntimes) {
  gc.beginCycle();
  EXPECT_TRUE(gc.pollInFlightBarriers()) << "nobody to wait for";
  gc.abortCycle();
}

TEST_F(SharedInFlightTest, Wait_SkipsAnIdleRuntime) {
  // A runtime outside every window cannot be acting on a stale test of the
  // flag, so it is not waited for, however long it stays idle.
  SharedMutator idle{gc, runtime};
  gc.beginCycle();
  EXPECT_TRUE(gc.pollInFlightBarriers()) << "an idle runtime was waited for";
  gc.abortCycle();
}

TEST_F(SharedInFlightTest, Wait_HoldsForAStoreThatTestedBeforeMarkingTurnedOn) {
  SharedMutator mutator{gc, runtime};

  // The store barrier's first steps: open the window and test the flag, which
  // says off, so nothing will be recorded.
  mutator.enterBarrier();
  EXPECT_EQ(0u, gc.cycleSeq() % 2);

  gc.beginCycle();
  EXPECT_FALSE(gc.pollInFlightBarriers())
      << "tracing could start with an unrecorded store still to land";

  // The store lands, and the window closes.
  mutator.exitBarrier();
  EXPECT_TRUE(gc.pollInFlightBarriers());
  gc.abortCycle();
}

TEST_F(SharedInFlightTest, Wait_IgnoresAnInnerExit) {
  // Windows nest, and only the outermost exit may end the wait: the outer
  // window's store is still to come.
  SharedMutator mutator{gc, runtime};
  mutator.enterBarrier();
  mutator.enterBarrier();

  gc.beginCycle();
  mutator.exitBarrier();
  EXPECT_FALSE(gc.pollInFlightBarriers()) << "an inner exit ended the wait";
  mutator.exitBarrier();
  EXPECT_TRUE(gc.pollInFlightBarriers());
  gc.abortCycle();
}

TEST_F(SharedInFlightTest, Wait_IsNotStarvedByABusyRuntime) {
  // A runtime found inside a new window has still closed the one being waited
  // for, and the new one tested the flag after marking turned on. Waiting for
  // an even count instead could wait forever on a runtime that is always in a
  // barrier when the collector looks.
  SharedMutator mutator{gc, runtime};
  mutator.enterBarrier();

  gc.beginCycle();
  mutator.exitBarrier();
  mutator.enterBarrier();
  EXPECT_TRUE(gc.pollInFlightBarriers())
      << "waited for the count to be even rather than for it to change";
  mutator.exitBarrier();
  gc.abortCycle();
}

TEST_F(SharedInFlightTest, Wait_WaitsForEveryRuntimeNotJustOne) {
  SharedMutator one{gc, runtime};
  SharedMutator two{gc, runtime};
  one.enterBarrier();
  two.enterBarrier();

  gc.beginCycle();
  one.exitBarrier();
  EXPECT_FALSE(gc.pollInFlightBarriers()) << "one runtime ended every wait";
  two.exitBarrier();
  EXPECT_TRUE(gc.pollInFlightBarriers());
  gc.abortCycle();
}

TEST_F(SharedInFlightTest, Wait_DoesNotReadARuntimeThatHasLeft) {
  // The collector holds no lock between polls, so a runtime it is waiting for
  // can leave in between. A poll that read its counter anyway would read freed
  // memory, which ASan catches here.
  SharedMutator resident{gc, runtime};
  auto leaving = std::make_unique<SharedMutator>(gc, runtime);
  resident.enterBarrier();
  leaving->enterBarrier();

  gc.beginCycle();
  leaving->exitBarrier();
  leaving.reset();
  EXPECT_FALSE(gc.pollInFlightBarriers()) << "resident is still inside";
  resident.exitBarrier();
  EXPECT_TRUE(gc.pollInFlightBarriers());
  gc.abortCycle();
}

TEST_F(
    SharedInFlightTest,
    Wait_IgnoresARuntimeThatAttachesAfterMarkingTurnsOn) {
  // Its registration is ordered after marking turned on, so even its first
  // window sees marking on, and waiting for it would only delay tracing. The
  // resident keeps the wait going, so the poll below does run.
  SharedMutator resident{gc, runtime};
  resident.enterBarrier();
  gc.beginCycle();

  SharedMutator latecomer{gc, runtime};
  latecomer.enterBarrier();
  resident.exitBarrier();
  EXPECT_TRUE(gc.pollInFlightBarriers())
      << "a runtime that attached after marking turned on was waited for";
  latecomer.exitBarrier();
  gc.abortCycle();
}

TEST_F(SharedInFlightTest, Wait_StartsAfreshEachCycle) {
  // What a cycle waits for is read when it begins, so an earlier cycle's wait
  // cannot satisfy a later one.
  SharedMutator mutator{gc, runtime};
  mutator.enterBarrier();
  gc.beginCycle();
  mutator.exitBarrier();
  EXPECT_TRUE(gc.pollInFlightBarriers());
  gc.abortCycle();

  mutator.enterBarrier();
  gc.beginCycle();
  EXPECT_FALSE(gc.pollInFlightBarriers())
      << "the previous cycle's wait satisfied this one";
  mutator.exitBarrier();
  EXPECT_TRUE(gc.pollInFlightBarriers());
  gc.abortCycle();
}

TEST_F(SharedInFlightTest, Set_OpensExactlyOneWindow) {
  SharedMutator mutator{gc, runtime};
  SharedArray *arr = SharedArray::create(gc, 1);
  ASSERT_NE(nullptr, arr);

  const uint32_t before = mutator.inFlightCounterSlow();
  arr->set(0, SharedValue::empty(), mutator);
  EXPECT_EQ(before + 2, mutator.inFlightCounterSlow())
      << "set() did not open and close exactly one window";

  // Inside another window, set()'s window is an inner one and leaves the
  // counter alone.
  mutator.enterBarrier();
  const uint32_t inside = mutator.inFlightCounterSlow();
  arr->set(0, SharedValue::empty(), mutator);
  EXPECT_EQ(inside, mutator.inFlightCounterSlow())
      << "an inner window wrote the counter";
  mutator.exitBarrier();
}

TEST_F(SharedInFlightTest, Wait_LetsAWaitedForRuntimeSpill) {
  // A runtime inside a window may fill its ring and spill, which takes
  // barrierMtx_. A wait that held that lock would deadlock with it, and this
  // test would hang.
  SharedArray *cell = SharedArray::create(gc, 0);
  ASSERT_NE(nullptr, cell);
  SharedMutator mutator{gc, runtime};
  std::atomic<bool> inside{false}, polling{false};

  std::thread runtimeThread([&] {
    mutator.enterBarrier();
    inside.store(true, std::memory_order_release);
    while (!polling.load(std::memory_order_acquire))
      std::this_thread::yield();
    const uint32_t seq = gc.cycleSeq();
    for (uint32_t i = 0; i < kRingCapacity * 2; ++i)
      mutator.push(cell, seq);
    mutator.exitBarrier();
  });

  while (!inside.load(std::memory_order_acquire))
    std::this_thread::yield();
  gc.beginCycle();
  // The runtime cannot leave its window before this, so the spills below all
  // happen while the collector is waiting.
  EXPECT_FALSE(gc.pollInFlightBarriers());
  polling.store(true, std::memory_order_release);
  while (!gc.pollInFlightBarriers())
    std::this_thread::yield();
  runtimeThread.join();
  EXPECT_FALSE(gc.barrierWorklistSlow().empty())
      << "the runtime never spilled during the wait";
  gc.abortCycle();
}

TEST_F(SharedInFlightTest, Wait_ReadsCountersAcrossThreadsSafely) {
  // Windows open and close on the runtime's thread while the collector reads
  // the counter on its own. Proves nothing single-threaded; it is here for
  // TSan.
  SharedMutator mutator{gc, runtime};
  std::atomic<bool> stop{false};
  std::atomic<uint32_t> windows{0};
  std::thread runtimeThread([&] {
    while (!stop.load(std::memory_order_acquire)) {
      mutator.enterBarrier();
      (void)gc.cycleSeq();
      mutator.exitBarrier();
      windows.fetch_add(1, std::memory_order_relaxed);
    }
  });

  // Not a fixed count alone: that can finish before the runtime's thread is
  // scheduled, leaving nothing for TSan to see.
  for (uint32_t i = 0;
       i < 200 || windows.load(std::memory_order_relaxed) < 1000;
       ++i) {
    gc.beginCycle();
    while (!gc.pollInFlightBarriers())
      std::this_thread::yield();
    gc.abortCycle();
  }
  stop.store(true, std::memory_order_release);
  runtimeThread.join();
}

} // namespace
