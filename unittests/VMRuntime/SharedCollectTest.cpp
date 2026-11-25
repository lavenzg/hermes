/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/BoxedSharedObject.h"
#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedMutator.h"

#include "VMRuntimeTestHelpers.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

namespace {

using namespace hermes;
using namespace hermes::vm;

constexpr size_t kAmpleBudget = 8 << 20;

/// Two runtimes, so "every attached runtime's roots" is a claim the fixture
/// can actually distinguish from "one runtime's roots, twice".
struct SharedCollectTest : public ::testing::Test {
  SharedGC gc{kAmpleBudget};
  std::shared_ptr<Runtime> rtA{Runtime::create(kTestRTConfig)};
  std::shared_ptr<Runtime> rtB{Runtime::create(kTestRTConfig)};
  Runtime &a{*rtA};
  Runtime &b{*rtB};

  ~SharedCollectTest() override {
    // The mutators are test locals and are already gone by here, so the
    // registry is empty and a cycle with no roots frees the heap -- which
    // ~SharedGC insists on. A legitimate collection rather than a teardown
    // special case.
    gc.collect();
  }

  SharedArray *array() {
    SharedArray *arr = SharedArray::create(gc, 1);
    EXPECT_NE(nullptr, arr);
    return arr;
  }

  void box(Runtime &runtime, SharedGCCell *cell) {
    GCScope scope{runtime, "SharedCollectTest"};
    auto handle = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
    runtime.setBoxForSharedCell(cell, handle);
  }
};

TEST_F(SharedCollectTest, Collect_KeepsWhatIsNamedAndFreesWhatIsNot) {
  // An end-to-end collection: turn marking on, wait, scan, trace, sweep.
  SharedMutator mutatorA{gc, a};
  SharedMutator mutatorB{gc, b};

  auto *namedByA = array();
  auto *namedByB = array();
  array(); // named by nobody
  box(a, namedByA);
  box(b, namedByB);
  ASSERT_EQ(3u, gc.liveCellCountSlow());

  gc.collect();

  EXPECT_EQ(2u, gc.liveCellCountSlow())
      << "the unnamed cell should have been the only casualty";
  // Reading these at all is the check: a swept cell's slot is poisoned, so
  // ASan traps here rather than returning a plausible value.
  EXPECT_EQ(1u, namedByA->length());
  EXPECT_EQ(1u, namedByB->length());
}

TEST_F(SharedCollectTest, Collect_ReachesThroughAnEdgeNoRuntimeNames) {
  // Tracing, not just root marking: only the parent is boxed, and the child
  // is reachable from it alone.
  SharedMutator mutatorA{gc, a};

  auto *parent = array();
  auto *child = array();
  // initField, not set: the graph is built before any cycle turns marking on,
  // so no barrier is owed.
  parent->initField(0, SharedValue::encodePointer(child));
  box(a, parent);

  gc.collect();

  EXPECT_EQ(2u, gc.liveCellCountSlow())
      << "the child was reachable only through the parent's slot";
}

TEST_F(SharedCollectTest, Collect_WaitsForABarrierInFlight) {
  // A barrier whose window opened before the cycle began records a cell well
  // after marking turned on. A collect() that did not wait would certify and
  // sweep first, and then drop the record as stale. Only certification finds
  // the cell, so a second round is needed to trace its child.
  SharedMutator mutatorA{gc, a};
  auto *rooted = array();
  auto *shaded = array();
  auto *child = array();
  shaded->initField(0, SharedValue::encodePointer(child));
  box(a, rooted);

  const uint32_t before = gc.cycleSeq();
  std::atomic<bool> inside{false};
  std::thread runtimeA([&] {
    mutatorA.enterBarrier();
    inside.store(true, std::memory_order_release);
    uint32_t seq;
    while ((seq = gc.cycleSeq()) == before)
      std::this_thread::yield();
    // Long enough for a collect() that does not wait to finish the cycle. One
    // that waits cannot get past this window, however long it stays open.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (seq % 2 == 1)
      mutatorA.push(shaded, seq);
    mutatorA.exitBarrier();
  });
  while (!inside.load(std::memory_order_acquire))
    std::this_thread::yield();
  gc.collect();
  runtimeA.join();

  EXPECT_EQ(3u, gc.liveCellCountSlow())
      << "a cell the barrier in flight recorded, or its child, was swept";
  EXPECT_EQ(1u, rooted->length());
  EXPECT_EQ(1u, shaded->length());
  EXPECT_EQ(1u, child->length());
}

TEST_F(SharedCollectTest, Collect_WaitsForAWindowOpenedDuringTheFirstWait) {
  // B's window holds collect() in its first wait. A, which the first pass
  // found idle, opens a window during that wait and records a cell only after
  // the wait has ended. With one pass, collect() would trace, certify and
  // sweep before that record lands; the second pass waits for A's window.
  SharedMutator mutatorA{gc, a};
  SharedMutator mutatorB{gc, b};
  auto *shaded = array(); // named by nothing but A's record

  mutatorB.enterBarrier();
  std::thread collector([&] { gc.collect(); });
  // The first pass reads every counter under one hold of barrierMtx_, so once
  // B is noted, A's counter has been read too, and found even.
  while (gc.awaitedCountSlow() != 1)
    std::this_thread::yield();
  mutatorA.enterBarrier();
  const uint32_t seq = gc.cycleSeq();
  mutatorB.exitBarrier();
  // Long enough for a collect() with one pass to finish the cycle.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  mutatorA.push(shaded, seq);
  mutatorA.exitBarrier();
  // Nothing may return before the join, or the thread is left running.
  collector.join();

  // Counted before `shaded` is read: it is alone in its segment, which the
  // sweep that freed it would also have unmapped.
  ASSERT_EQ(1u, gc.liveCellCountSlow()) << "the cell A recorded was swept";
  EXPECT_EQ(1u, shaded->length());
}

TEST_F(SharedCollectTest, Collect_FreesEverythingWithNoRuntimes) {
  // No registry, so no roots. Everything allocated is garbage, and this is
  // also the shape the teardown helper in the other suites relies on.
  array();
  array();
  ASSERT_EQ(2u, gc.liveCellCountSlow());

  gc.collect();

  EXPECT_EQ(0u, gc.liveCellCountSlow());
}

TEST_F(SharedCollectTest, Collect_LeavesTheHeapCollectableAgain) {
  // Two cycles back to back. The second is what catches a cycle that ends in
  // a state the next one cannot start from -- a stale mark, a live MarkState,
  // parity left inverted.
  SharedMutator mutatorA{gc, a};

  auto *kept = array();
  box(a, kept);
  array(); // garbage

  gc.collect();
  ASSERT_EQ(1u, gc.liveCellCountSlow());

  array(); // more garbage, born after the first cycle
  ASSERT_EQ(2u, gc.liveCellCountSlow());

  gc.collect();
  EXPECT_EQ(1u, gc.liveCellCountSlow());
  EXPECT_EQ(1u, kept->length()) << "a survivor of two cycles";
}

TEST_F(SharedCollectTest, Collect_ReturnsTheHeapToTheOS) {
  // sweep()'s return value, threaded out through collect(). Enough cells to
  // fill whole segments, so releaseRegion actually runs.
  for (uint32_t i = 0; i < 4096; ++i)
    array();
  const size_t before = gc.footprint();
  ASSERT_GT(before, 0u);

  const size_t released = gc.collect();

  EXPECT_GT(released, 0u) << "no segment was returned";
  EXPECT_LT(gc.footprint(), before);
  EXPECT_EQ(0u, gc.liveCellCountSlow());
}

} // namespace
