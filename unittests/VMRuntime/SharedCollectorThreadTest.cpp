/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/BoxedSharedObject.h"
#include "hermes/VM/JSArray.h"
#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedGCBoxing-inline.h"
#include "hermes/VM/SharedHeap/SharedMutator.h"

#include "VMRuntimeTestHelpers.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>

namespace {

using namespace hermes;
using namespace hermes::vm;
using std::chrono::milliseconds;

constexpr size_t kBudget = 8 << 20;

/// Long enough that the array gets a region of its own, one segment.
constexpr uint32_t kOversizedLength =
    SharedGC::SizeBuckets::kMaxBucketSize / sizeof(SharedValue) + 1;

/// A cycle once the footprint grows by four regions, none by time, and no
/// minimum gap, so only the last cycle's length spaces cycles.
SharedGC::CollectorConfig quickConfig() {
  SharedGC::CollectorConfig config;
  config.minTriggerBytes = 4 * SharedGC::kSegmentSize;
  config.minGap = milliseconds(0);
  config.maxGap = std::chrono::hours(1);
  return config;
}

/// Poll \p done until it holds or \p timeout passes. The collector thread runs
/// on its own schedule, which the sanitizers slow down a lot.
template <typename Done>
bool waitFor(Done done, milliseconds timeout = std::chrono::seconds(60)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() > deadline)
      return false;
    std::this_thread::sleep_for(milliseconds(1));
  }
  return true;
}

/// Allocate enough unnamed arrays, each a region of its own, to take the
/// footprint to \p bytes. The count is fixed first: a cycle that runs in the
/// meantime lowers the footprint, and allocating on would leave arrays that
/// the cycle never saw.
void allocateGarbageTo(SharedGC &gc, size_t bytes) {
  const size_t now = gc.footprint();
  const size_t count = bytes > now
      ? (bytes - now + SharedGC::kSegmentSize - 1) / SharedGC::kSegmentSize
      : 0;
  for (size_t i = 0; i < count; ++i)
    ASSERT_NE(nullptr, SharedArray::create(gc, kOversizedLength));
}

/// Box arrays in \p rt, each a region of its own, until the footprint reaches
/// \p bytes. Every box goes into \p boxes, so every array stays named.
void boxArraysTo(
    SharedGC &gc,
    Runtime &rt,
    Handle<JSArray> boxes,
    size_t bytes) {
  struct : public Locals {
    PinnedValue<BoxedSharedObject> box;
  } lv;
  LocalsRAII lraii{rt, &lv};
  while (gc.footprint() < bytes) {
    lv.box = gc.makeBoxed<SharedArray>(
        rt, SharedArray::allocationSize(kOversizedLength), kOversizedLength);
    ASSERT_TRUE(*lv.box);
    ASSERT_NE(
        ExecutionStatus::EXCEPTION,
        JSArray::setElementAt(boxes, rt, boxes->getEndIndex(), lv.box));
  }
}

TEST(SharedCollectorThreadTest, Thread_StopsWithTheHeap) {
  // Give the thread time to fall asleep. Nothing is due, so it sleeps with no
  // deadline, and the destructor must still wake it and join it.
  SharedGC gc{kBudget, quickConfig()};
  std::this_thread::sleep_for(milliseconds(50));
  EXPECT_EQ(0u, gc.cycleCountSlow());
}

TEST(
    SharedCollectorThreadTest,
    Trigger_IsTwiceWhatTheSweepLeftCappedHalfwayToTheBudget) {
  auto heap = std::make_shared<SharedGC>(kBudget, quickConfig());
  SharedGC &gc = *heap;
  // Nothing left yet: minTriggerBytes, below half the budget.
  EXPECT_EQ(4 * SharedGC::kSegmentSize, gc.triggerBytesSlow());

  auto rt = Runtime::create(kTestRTConfig, heap);
  struct : public Locals {
    PinnedValue<JSArray> boxes;
  } lv;
  LocalsRAII lraii{*rt, &lv};
  auto created = JSArray::create(*rt, 0, 0);
  ASSERT_NE(ExecutionStatus::EXCEPTION, created.getStatus());
  lv.boxes = std::move(*created);
  // 2 MiB left: twice that, 4 MiB, is below halfway to the budget, 5 MiB.
  boxArraysTo(gc, *rt, lv.boxes, 2 << 20);
  gc.collect();
  EXPECT_EQ(size_t{2} << 20, gc.footprint());
  EXPECT_EQ(size_t{4} << 20, gc.triggerBytesSlow());
  // 6 MiB left: halfway to the budget, 7 MiB, is below twice that.
  boxArraysTo(gc, *rt, lv.boxes, 6 << 20);
  gc.collect();
  EXPECT_EQ(size_t{6} << 20, gc.footprint());
  EXPECT_EQ(size_t{7} << 20, gc.triggerBytesSlow());
}

TEST(SharedCollectorThreadTest, Trigger_IsHalfwayForAHugeMinimumGrowth) {
  // The sweep left plus minTriggerBytes saturates rather than wrapping on a
  // 64-bit host, so the halfway cap is what applies.
  SharedGC::CollectorConfig config = quickConfig();
  config.minTriggerBytes = std::numeric_limits<size_t>::max();
  auto heap = std::make_shared<SharedGC>(kBudget, config);
  SharedGC &gc = *heap;
  auto rt = Runtime::create(kTestRTConfig, heap);
  struct : public Locals {
    PinnedValue<JSArray> boxes;
  } lv;
  LocalsRAII lraii{*rt, &lv};
  auto created = JSArray::create(*rt, 0, 0);
  ASSERT_NE(ExecutionStatus::EXCEPTION, created.getStatus());
  lv.boxes = std::move(*created);
  boxArraysTo(gc, *rt, lv.boxes, 2 << 20);
  gc.collect();
  EXPECT_EQ(size_t{2} << 20, gc.footprint());
  // Halfway from 2 MiB to the 8 MiB budget.
  EXPECT_EQ(size_t{5} << 20, gc.triggerBytesSlow());
}

TEST(SharedCollectorThreadTest, Trigger_IsOffWhenTheSweepLeavesNoRoom) {
  // Live arrays fill the whole budget. Halfway from the footprint to the
  // budget is no growth at all, so the footprint must not make a cycle due:
  // it would stay due, and the thread would collect back to back while
  // nothing happens.
  constexpr size_t kSmallBudget = 8 * SharedGC::kSegmentSize;
  auto heap = std::make_shared<SharedGC>(kSmallBudget, quickConfig());
  SharedGC &gc = *heap;
  auto rt = Runtime::create(kTestRTConfig, heap);
  struct : public Locals {
    PinnedValue<JSArray> boxes;
  } lv;
  LocalsRAII lraii{*rt, &lv};
  auto created = JSArray::create(*rt, 0, 0);
  ASSERT_NE(ExecutionStatus::EXCEPTION, created.getStatus());
  lv.boxes = std::move(*created);
  boxArraysTo(gc, *rt, lv.boxes, kSmallBudget);
  gc.collect();
  EXPECT_EQ(kSmallBudget, gc.footprint());
  EXPECT_EQ(std::numeric_limits<size_t>::max(), gc.triggerBytesSlow());
  // A cycle the thread found due while the heap filled may run once more,
  // after the one above; let it finish before counting.
  std::this_thread::sleep_for(milliseconds(200));
  const uint64_t cycles = gc.cycleCountSlow();
  std::this_thread::sleep_for(milliseconds(200));
  EXPECT_EQ(cycles, gc.cycleCountSlow()) << "a full heap kept a cycle due";
}

TEST(SharedCollectorThreadTest, Footprint_StartsACycleOnceItReachesTheTrigger) {
  SharedGC gc{kBudget, quickConfig()};
  allocateGarbageTo(gc, gc.triggerBytesSlow());
  ASSERT_TRUE(waitFor([&] { return gc.cycleCountSlow() >= 1; }));
  // Every array predates the cycle and none is named.
  EXPECT_EQ(0u, gc.liveCellCountSlow());
}

TEST(SharedCollectorThreadTest, Footprint_CyclesKeepWhatARuntimeNames) {
  auto heap = std::make_shared<SharedGC>(kBudget, quickConfig());
  SharedGC &gc = *heap;
  auto rt = Runtime::create(kTestRTConfig, heap);
  struct : public Locals {
    PinnedValue<BoxedSharedObject> box;
  } lv;
  LocalsRAII lraii{*rt, &lv};
  lv.box = gc.makeBoxed<SharedArray>(
      *rt, SharedArray::allocationSize(1), uint32_t{1});
  ASSERT_TRUE(*lv.box);
  auto *named = vmcast<SharedArray>(lv.box->getSharedCell());

  allocateGarbageTo(gc, gc.triggerBytesSlow());
  ASSERT_TRUE(waitFor([&] { return gc.cycleCountSlow() >= 1; }));
  // Counted before `named` is read: a swept cell's slot is poisoned.
  ASSERT_EQ(1u, gc.liveCellCountSlow()) << "the named array was swept";
  EXPECT_EQ(1u, named->length());
}

TEST(SharedCollectorThreadTest, Exhaustion_WakesTheCollector) {
  // The footprint stays below the trigger, so only the refusal makes a cycle
  // due.
  constexpr size_t kSmallBudget = 16 * SharedGC::kSegmentSize;
  SharedGC::CollectorConfig config = quickConfig();
  config.minTriggerBytes = kSmallBudget;
  SharedGC gc{kSmallBudget, config};
  ASSERT_NE(nullptr, SharedArray::create(gc, 1));
  EXPECT_LT(gc.footprint(), gc.triggerBytesSlow());

  // Fifteen and a half segments of elements need a region of all sixteen,
  // which the budget refuses while the first array's segment is mapped.
  const uint32_t wholeBudget = static_cast<uint32_t>(
      (15 * SharedGC::kSegmentSize + SharedGC::kSegmentSize / 2) /
      sizeof(SharedValue));
  EXPECT_EQ(0u, gc.cycleCountSlow());
  EXPECT_EQ(nullptr, SharedArray::create(gc, wholeBudget));
  ASSERT_TRUE(waitFor([&] { return gc.cycleCountSlow() >= 1; }));
  // That cycle cleared the refusal, and nothing has happened since.
  std::this_thread::sleep_for(milliseconds(200));
  EXPECT_EQ(1u, gc.cycleCountSlow()) << "one refusal made more than one cycle";
  EXPECT_NE(nullptr, SharedArray::create(gc, wholeBudget))
      << "the cycle should have freed the first array";
}

TEST(SharedCollectorThreadTest, Timer_NeverRunsACycleInAHeapNothingChanges) {
  SharedGC::CollectorConfig config = quickConfig();
  config.maxGap = milliseconds(10);
  SharedGC gc{kBudget, config};
  std::this_thread::sleep_for(milliseconds(200));
  EXPECT_EQ(0u, gc.cycleCountSlow());
}

TEST(
    SharedCollectorThreadTest,
    Timer_RunsACycleMaxGapAfterTheLastIfAnythingWasAllocated) {
  SharedGC::CollectorConfig config = quickConfig();
  config.maxGap = milliseconds(100);
  const auto start = std::chrono::steady_clock::now();
  SharedGC gc{kBudget, config};
  // Far below the footprint trigger.
  ASSERT_NE(nullptr, SharedArray::create(gc, 1));
  ASSERT_TRUE(waitFor([&] { return gc.cycleCountSlow() >= 1; }));
  EXPECT_GE(std::chrono::steady_clock::now() - start, config.maxGap)
      << "the cycle came before maxGap had passed";
  // Nothing allocated since that cycle began, so no other is due.
  std::this_thread::sleep_for(3 * config.maxGap);
  EXPECT_EQ(1u, gc.cycleCountSlow());
}

TEST(SharedCollectorThreadTest, Timer_RunsACycleAfterARuntimeLeaves) {
  // The cells only a leaving runtime named become garbage without any
  // allocation, so its leaving turns on the timer too.
  SharedGC::CollectorConfig config = quickConfig();
  config.maxGap = milliseconds(100);
  auto heap = std::make_shared<SharedGC>(kBudget, config);
  SharedGC &gc = *heap;
  auto rt = Runtime::create(kTestRTConfig, heap);
  {
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{*rt, &lv};
    lv.box = gc.makeBoxed<SharedArray>(
        *rt, SharedArray::allocationSize(1), uint32_t{1});
    ASSERT_TRUE(*lv.box);
  }
  // The allocation turned the timer on once; that cycle keeps the boxed array.
  ASSERT_TRUE(waitFor([&] { return gc.cycleCountSlow() >= 1; }));
  EXPECT_EQ(1u, gc.liveCellCountSlow());

  rt.reset();
  ASSERT_TRUE(waitFor([&] { return gc.cycleCountSlow() >= 2; }));
  EXPECT_EQ(0u, gc.liveCellCountSlow());
}

TEST(SharedCollectorThreadTest, Gap_SpacesCyclesAtLeastMinGapApart) {
  SharedGC::CollectorConfig config = quickConfig();
  config.minGap = milliseconds(200);
  SharedGC gc{kBudget, config};
  // Garbage for a second keeps a cycle due almost all the time; an
  // allocation the budget refuses only makes it due again.
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(1))
    SharedArray::create(gc, kOversizedLength);
  const uint64_t cycles = gc.cycleCountSlow();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_GE(cycles, 1u);
  // The first may start at once and each later one no sooner than minGap
  // after the last ended.
  EXPECT_LE(cycles, 1 + static_cast<uint64_t>(elapsed / config.minGap));
}

TEST(SharedCollectorThreadTest, Gap_IsAtLeastAsLongAsTheLastCycle) {
  // With no minimum gap, only the last cycle's length spaces cycles. A window
  // held open for two seconds stretches one cycle, whose in-flight wait waits
  // for it, to at least that long.
  SharedGC gc{kBudget, quickConfig()};
  auto rt = Runtime::create(kTestRTConfig);
  SharedMutator mutator{gc, *rt};
  mutator.enterBarrier();
  allocateGarbageTo(gc, gc.triggerBytesSlow());
  // Closed before anything can return early: ~SharedMutator asserts that no
  // window is open.
  const bool waited = waitFor([&] { return gc.awaitedCountSlow() == 1; });
  if (waited)
    std::this_thread::sleep_for(std::chrono::seconds(2));
  mutator.exitBarrier();
  ASSERT_TRUE(waited) << "the cycle never waited for the window";
  ASSERT_TRUE(waitFor([&] { return gc.cycleCountSlow() >= 1; }));
  const auto firstEnded = std::chrono::steady_clock::now();

  // A cycle is due again at once, but may not start for two more seconds.
  allocateGarbageTo(gc, gc.triggerBytesSlow());
  std::this_thread::sleep_until(firstEnded + milliseconds(300));
  EXPECT_EQ(1u, gc.cycleCountSlow()) << "the gap was shorter than the cycle";
}

TEST(SharedCollectorThreadTest, Collect_TakesTurnsWithTheThread) {
  // An allocator keeps the thread's trigger firing while this thread runs
  // cycles of its own. Two cycles at once would trip beginCycle's assert that
  // none is in progress.
  SharedGC gc{kBudget, quickConfig()};
  std::atomic<bool> done{false};
  std::thread allocator([&] {
    while (!done.load(std::memory_order_acquire))
      SharedArray::create(gc, kOversizedLength);
  });
  for (int i = 0; i < 50; ++i)
    gc.collect();
  done.store(true, std::memory_order_release);
  allocator.join();
  EXPECT_GE(gc.cycleCountSlow(), 50u);
}

TEST(
    SharedCollectorThreadTest,
    Concurrent_RuntimesStoreAndMaterializeWhileTheThreadCollects) {
  // A writer moves the only reference to `treasure` between two arrays that A
  // boxes, and allocates garbage that keeps the trigger firing. A reader
  // materialises both slots in B. Meanwhile the collector thread runs cycles
  // on its own, and where the post-certification check is built, each one
  // checks that nothing reachable was left unmarked.
  auto heap = std::make_shared<SharedGC>(kBudget, quickConfig());
  SharedGC &gc = *heap;
  // Created here rather than on the threads that use them: creating two
  // runtimes at once races inside libc's tzset, which ThreadSanitizer reports.
  auto rtA = Runtime::create(kTestRTConfig, heap);
  auto rtB = Runtime::create(kTestRTConfig, heap);
  auto rtW = Runtime::create(kTestRTConfig);
  SharedMutator writerMutator{gc, *rtW};

  struct : public Locals {
    PinnedValue<BoxedSharedObject> src;
    PinnedValue<BoxedSharedObject> dst;
  } lvA;
  LocalsRAII lraiiA{*rtA, &lvA};
  lvA.src = gc.makeBoxed<SharedArray>(
      *rtA, SharedArray::allocationSize(1), uint32_t{1});
  lvA.dst = gc.makeBoxed<SharedArray>(
      *rtA, SharedArray::allocationSize(1), uint32_t{1});
  ASSERT_TRUE(*lvA.src && *lvA.dst);
  auto *src = vmcast<SharedArray>(lvA.src->getSharedCell());
  auto *dst = vmcast<SharedArray>(lvA.dst->getSharedCell());
  SharedArray *treasure = SharedArray::create(gc, 1);
  ASSERT_NE(nullptr, treasure);
  // A barriered store: `src` is in A's map, and the collector thread can trace
  // it at any time.
  src->set(0, SharedValue::encodePointer(treasure), writerMutator);

  std::atomic<bool> done{false};
  std::atomic<uint32_t> moves{0};
  std::atomic<uint32_t> boxed{0};
  std::thread writer([&] {
    while (!done.load(std::memory_order_acquire)) {
      SharedArray *from = src, *to = dst;
      if (from->at(0).isEmpty())
        std::swap(from, to);
      const SharedValue held = from->at(0);
      if (held.isEmpty())
        continue;
      // Destination first, so the reference is never in neither slot.
      to->set(0, held, writerMutator);
      from->set(0, SharedValue::empty(), writerMutator);
      SharedArray::create(gc, kOversizedLength);
      moves.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::thread reader([&] {
    Runtime &b = *rtB;
    struct : public Locals {
      PinnedValue<BoxedSharedObject> box;
    } lv;
    LocalsRAII lraii{b, &lv};
    for (uint32_t i = 0; !done.load(std::memory_order_acquire); ++i) {
      SharedArray *holder = i % 2 ? src : dst;
      lv.box = b.materializeSharedSlot(*holder->begin());
      if (*lv.box)
        boxed.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  done.store(true, std::memory_order_release);
  writer.join();
  reader.join();

  EXPECT_GT(gc.cycleCountSlow(), 0u) << "the thread ran no cycle";
  gc.collect();
  const SharedValue held = src->at(0).isEmpty() ? dst->at(0) : src->at(0);
  EXPECT_EQ(treasure, held.getPointerOrNull());
  // ASan traps this read if `treasure` was swept: a freed slot is poisoned.
  EXPECT_EQ(1u, treasure->length());
  EXPECT_GT(moves.load(), 0u);
  EXPECT_GT(boxed.load(), 0u);
}

} // namespace
