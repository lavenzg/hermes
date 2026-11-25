/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/SharedHeap/SharedBarrierRing.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

using namespace hermes;
using namespace hermes::vm;

constexpr uint32_t kCap = SharedBarrierRing::kCapacity;
constexpr uint32_t kSeq = 7;

/// Fabricated addresses, not real cells: the ring never dereferences what it
/// carries, and a real heap would blur "the ring kept the bits" with "the heap
/// happened to still be valid".
SharedGCCell *cellFor(uint32_t i) {
  return reinterpret_cast<SharedGCCell *>(uintptr_t{0x1000} + i * 8);
}

/// Push \p n entries, spilling into \p out whenever the ring fills, exactly as
/// SharedMutator::push will.
void pushN(
    SharedBarrierRing &ring,
    uint32_t n,
    std::vector<SharedGCCell *> &out,
    uint32_t seq = kSeq) {
  for (uint32_t i = 0; i < n; ++i) {
    if (ring.full()) {
      ring.drainInto(out, seq);
    }
    ring.push(cellFor(i), seq);
  }
}

TEST(SharedBarrierRingTest, Drain_YieldsEntriesOldestFirst) {
  SharedBarrierRing ring;
  for (uint32_t i = 0; i < 4; ++i)
    ring.push(cellFor(i), kSeq);

  std::vector<SharedGCCell *> out;
  ring.drainInto(out, kSeq);

  ASSERT_EQ(4u, out.size());
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_EQ(cellFor(i), out[i]) << "entry " << i;
}

TEST(SharedBarrierRingTest, Drain_LeavesTheRingEmpty) {
  SharedBarrierRing ring;
  ring.push(cellFor(0), kSeq);
  std::vector<SharedGCCell *> out;
  ring.drainInto(out, kSeq);

  EXPECT_EQ(0u, ring.sizeSlow());
  out.clear();
  ring.drainInto(out, kSeq);
  EXPECT_TRUE(out.empty());
}

TEST(SharedBarrierRingTest, Drain_DropsEntriesFromAnotherCycle) {
  // Why Entry carries a stamp: a cell pushed in an earlier cycle may since
  // have been swept and its segments unmapped.
  SharedBarrierRing ring;
  ring.push(cellFor(0), kSeq - 1);
  ring.push(cellFor(1), kSeq);
  ring.push(cellFor(2), kSeq + 1);

  std::vector<SharedGCCell *> out;
  ring.drainInto(out, kSeq);

  ASSERT_EQ(1u, out.size());
  EXPECT_EQ(cellFor(1), out[0]);
  EXPECT_EQ(0u, ring.sizeSlow()) << "a dropped entry is still consumed";
}

TEST(SharedBarrierRingTest, Full_ReportsExactlyAtCapacity) {
  SharedBarrierRing ring;
  for (uint32_t i = 0; i < kCap; ++i) {
    ASSERT_FALSE(ring.full()) << "at " << i;
    ring.push(cellFor(i), kSeq);
  }
  EXPECT_TRUE(ring.full());
}

TEST(SharedBarrierRingTest, DiscardAll_ConsumesWithoutReading) {
  // What a spill with no cycle in progress uses: the ring holds a past
  // cycle's entries, whose cells may already be freed. These bogus pointers
  // would fault if anything read them.
  SharedBarrierRing ring;
  for (uint32_t i = 0; i < 8; ++i)
    ring.push(
        reinterpret_cast<SharedGCCell *>(uintptr_t{0xdead0000} + i), kSeq);

  ring.discardAll();

  EXPECT_EQ(0u, ring.sizeSlow());
  EXPECT_FALSE(ring.full());
  std::vector<SharedGCCell *> out;
  ring.drainInto(out, kSeq);
  EXPECT_TRUE(out.empty());
}

TEST(SharedBarrierRingTest, KeepsSpillingAfterTheFirstSpill) {
  // The producer's cached head goes stale every time it drains its own ring,
  // and only full()'s slow path recovers. One spill would pass either way.
  constexpr uint32_t kTotal = kCap * 16 + 5;
  SharedBarrierRing ring;
  std::vector<SharedGCCell *> spilled;
  pushN(ring, kTotal, spilled);

  std::vector<SharedGCCell *> rest;
  ring.drainInto(rest, kSeq);

  // A full ring every kCap pushes. A stale cached head would read as full at
  // every push after the first wrap and spill one entry at a time.
  EXPECT_EQ(kCap * 16, spilled.size());
  EXPECT_EQ(5u, rest.size());
}

TEST(SharedBarrierRingTest, Wraps_WithoutLosingOrReorderingEntries) {
  // Drive the cursors well past kCapacity, draining a prime number at a time
  // so drains do not line up with the wrap.
  SharedBarrierRing ring;
  std::vector<SharedGCCell *> got;
  uint32_t pushed = 0;
  for (uint32_t round = 0; round < 40; ++round) {
    for (uint32_t i = 0; i < 13; ++i) {
      ASSERT_FALSE(ring.full());
      ring.push(cellFor(pushed++), kSeq);
    }
    ring.drainInto(got, kSeq);
  }

  ASSERT_EQ(pushed, got.size());
  for (uint32_t i = 0; i < pushed; ++i)
    ASSERT_EQ(cellFor(i), got[i]) << "entry " << i;
}

TEST(SharedBarrierRingTest, ConcurrentProducerAndConsumer_LoseNothing) {
  // The collector steals while the producer is still pushing. Postcondition is
  // schedule-independent: every entry appears exactly once.
  constexpr uint32_t kTotal = 100000;
  SharedBarrierRing ring;
  std::vector<SharedGCCell *> stolen;
  std::atomic<bool> done{false};
  std::atomic<uint32_t> produced{0};

  std::thread consumer([&] {
    while (!done.load(std::memory_order_acquire))
      ring.drainInto(stolen, kSeq);
    ring.drainInto(stolen, kSeq);
  });

  std::vector<SharedGCCell *> selfSpilled;
  for (uint32_t i = 0; i < kTotal; ++i) {
    // Spin rather than drain here: draining would make two consumers race,
    // which is what barrierMtx_ prevents in the real thing.
    while (ring.full())
      std::this_thread::yield();
    ring.push(cellFor(i), kSeq);
    produced.fetch_add(1, std::memory_order_relaxed);
  }
  done.store(true, std::memory_order_release);
  consumer.join();

  ASSERT_EQ(kTotal, produced.load());
  ASSERT_EQ(kTotal, stolen.size() + selfSpilled.size())
      << "an entry was lost or duplicated";
  std::unordered_set<const void *> seen;
  for (SharedGCCell *c : stolen)
    EXPECT_TRUE(seen.insert(c).second) << "duplicate entry";
}

} // namespace
