/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGC-inline.h"
#include "hermes/VM/SharedHeap/SharedMutator.h"
#include "hermes/VM/SharedHeap/SharedValue.h"

#include "VMRuntimeTestHelpers.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

using namespace hermes;
using namespace hermes::vm;

/// Ample enough that budget exhaustion is never the thing under test.
constexpr size_t kAmpleBudget = 8 << 20;

/// The length of the array the writer builds. Not 0 or 1, so a length read
/// from memory the build never wrote is unlikely to match it.
constexpr uint32_t kBuiltLength = 3;

/// What a reader saw of the cell it reached through the slot.
struct Seen {
  CellKind kind = CellKind::UninitializedKind;
  uint32_t length = 0;

  void read(SharedGCCell *cell) {
    kind = cell->getKind();
    length = static_cast<SharedArray *>(cell)->length();
  }
};

/// Hands each cell straight to a Seen. The collector's own acceptor reads only
/// the cell's mark bit, which needs the same acquire -- a cell built after its
/// slot was traced must be seen marked -- but the bit is atomic, so a stale
/// read of it is no data race. Reading the plain kind and length instead is
/// what lets ThreadSanitizer see a missing acquire.
struct ReadingAcceptor {
  Seen &seen;
  bool found = false;

  void accept(SharedGCCell *cell) {
    seen.read(cell);
    found = true;
  }
};

/// Each test publishes one newly built array from a writer thread to a reader
/// through a slot, and nothing else orders the two: no lock they share, no
/// other atomic, and no gtest macro inside either thread, since a failing one
/// takes a lock. SharedArray::create writes the array's kind and length with
/// plain stores, so the reader's reads of them are ordered after the build
/// only by the slot store's release and the read's acquire. Weaken either to
/// relaxed and ThreadSanitizer flags a data race on those fields; no other
/// build can see the difference.
struct SharedValueTest : public RuntimeTestFixture {
  SharedGC gc{kAmpleBudget};

  ~SharedValueTest() override {
    // A rootless cycle, not a bare sweep: see the other suites' freeAll.
    gc.beginCycle();
    while (!gc.certifyRound())
      gc.drainMarkWorklist();
    gc.endCycle();
    gc.sweep();
  }

  /// Build an array of kBuiltLength on a writer thread and store it into
  /// \p slot of \p holder, while \p read runs on a reader thread, until both
  /// are done.
  template <typename Read>
  void publish(SharedArray *holder, uint32_t slot, Read read) {
    std::thread reader(read);
    std::thread writer([&] {
      SharedMutator mutator{gc, runtime};
      SharedArray *built = SharedArray::create(gc, kBuiltLength);
      holder->set(slot, SharedValue::encodePointer(built), mutator);
    });
    writer.join();
    reader.join();
  }
};

TEST_F(SharedValueTest, Concurrent_ACopySeesTheCellAsBuilt) {
  // The route a runtime takes: SharedArray::at copies the slot, and the copy
  // constructor's load is the acquire.
  auto *holder = SharedArray::create(gc, 1);
  ASSERT_NE(nullptr, holder);
  Seen seen;
  publish(holder, 0, [&] {
    for (;;) {
      const SharedValue copy = holder->at(0);
      if (copy.isPointer()) {
        seen.read(copy.getPointer());
        return;
      }
      std::this_thread::yield();
    }
  });

  EXPECT_EQ(CellKind::SharedArrayKind, seen.kind);
  EXPECT_EQ(kBuiltLength, seen.length);
}

TEST_F(SharedValueTest, Concurrent_GetPointerSeesTheCellAsBuilt) {
  // The slot read in place, with no copy. isPointer's load is relaxed, so
  // getPointer's load is the acquire.
  auto *holder = SharedArray::create(gc, 1);
  ASSERT_NE(nullptr, holder);
  Seen seen;
  publish(holder, 0, [&] {
    const SharedValue &slot = *holder->begin();
    while (!slot.isPointer())
      std::this_thread::yield();
    seen.read(slot.getPointer());
  });

  EXPECT_EQ(CellKind::SharedArrayKind, seen.kind);
  EXPECT_EQ(kBuiltLength, seen.length);
}

TEST_F(SharedValueTest, Concurrent_TheTracerSeesTheCellAsBuilt) {
  // The collector's route: markCell decodes each slot with getPointerOrNull,
  // whose load is the acquire.
  auto *holder = SharedArray::create(gc, 1);
  ASSERT_NE(nullptr, holder);
  Seen seen;
  publish(holder, 0, [&] {
    ReadingAcceptor acceptor{seen};
    while (!acceptor.found) {
      SharedGC::markCell(acceptor, holder);
      std::this_thread::yield();
    }
  });

  EXPECT_EQ(CellKind::SharedArrayKind, seen.kind);
  EXPECT_EQ(kBuiltLength, seen.length);
}

TEST_F(SharedValueTest, Concurrent_TheBarrierSeesTheCellAsBuilt) {
  // The store barrier's route. With marking on, a second writer's pre-read
  // loads the new array and pushes it; when that writer leaves, its ring moves
  // to the barrier worklist, where the collector reads it. The pre-read's load
  // is the acquire. The first writer must not leave until the array has been
  // read: leaving takes barrierMtx_, and that lock alone would order the build
  // before the read. No ASSERT between beginCycle and endCycle, since an early
  // return would leave the cycle open.
  auto *holder = SharedArray::create(gc, 1);
  ASSERT_NE(nullptr, holder);
  gc.beginCycle();
  std::atomic<bool> read{false};
  std::thread writer([&] {
    SharedMutator mutator{gc, runtime};
    SharedArray *built = SharedArray::create(gc, kBuiltLength);
    holder->set(0, SharedValue::encodePointer(built), mutator);
    while (!read.load(std::memory_order_acquire))
      std::this_thread::yield();
  });
  std::thread overwriter([&] {
    SharedMutator mutator{gc, runtime};
    const SharedValue &slot = *holder->begin();
    while (!slot.isPointer())
      std::this_thread::yield();
    holder->set(0, SharedValue::empty(), mutator);
  });
  overwriter.join();
  const std::vector<SharedGCCell *> recorded = gc.barrierWorklistSlow();
  Seen seen;
  if (recorded.size() == 1)
    seen.read(recorded[0]);
  read.store(true, std::memory_order_release);
  writer.join();
  while (!gc.certifyRound())
    gc.drainMarkWorklist();
  gc.endCycle();

  EXPECT_EQ(1u, recorded.size()) << "the pre-read recorded nothing";
  EXPECT_EQ(CellKind::SharedArrayKind, seen.kind);
  EXPECT_EQ(kBuiltLength, seen.length);
}

} // namespace
