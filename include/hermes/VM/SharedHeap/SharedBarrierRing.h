/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/Support/Compiler.h"
#include "hermes/VM/SharedHeap/SharedGCCell.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <vector>

namespace hermes {
namespace vm {

/// One runtime's local barrier list.
///
/// Single producer, single consumer. The producer is the thread owning the
/// runtime, and is also one of the two consumers -- itself when it spills, the
/// collector when it steals -- which SharedGC's barrier mutex keeps from
/// overlapping. The producer takes that mutex only to spill, never while
/// pushing, so what orders slot reuse is the release/acquire pair on head_
/// rather than the lock.
class SharedBarrierRing {
 public:
  /// Entries buffered before the barrier spills. A power of two, so the
  /// monotone cursors below mask to an index.
  static constexpr uint32_t kCapacity = 256;

  /// A shaded cell, plus the cycle number its barrier read. An entry stamped
  /// with an older cycle may name a swept cell, so it has to be dropped
  /// without ever being dereferenced.
  struct Entry {
    SharedGCCell *cell;
    uint32_t seq;
  };

  SharedBarrierRing() = default;

  SharedBarrierRing(const SharedBarrierRing &) = delete;
  SharedBarrierRing &operator=(const SharedBarrierRing &) = delete;

  /// Whether the next push would overwrite an unconsumed entry. Re-reads head_
  /// only on the slow path, which is also what recovers a cache left stale by
  /// the producer draining its own ring.
  /// \pre the caller is this ring's producer.
  bool full() {
    const uint32_t t = tail_.load(std::memory_order_relaxed);
    if (LLVM_LIKELY(t - headCache_ < kCapacity))
      return false;
    headCache_ = head_.load(std::memory_order_acquire);
    return t - headCache_ >= kCapacity;
  }

  /// Append \p cell, stamped \p seq.
  /// \pre !full(), and the caller is this ring's producer.
  void push(SharedGCCell *cell, uint32_t seq) {
    assert(!reentered() && "two producers on one ring");
    const uint32_t t = tail_.load(std::memory_order_relaxed);
    // Must stay relaxed: an acquire here gives debug builds an ordering edge
    // release builds lack, which hides slot-reuse races from TSan.
    assert(
        t - head_.load(std::memory_order_relaxed) < kCapacity &&
        "push into a full ring");
    entries_[t & (kCapacity - 1)] = Entry{cell, seq};
    // Publishes the entry. This is not the barrier's ordering edge -- that has
    // to be the slot store the barrier guards, a release that carries the push
    // to whoever reads the value it stores.
    tail_.store(t + 1, std::memory_order_release);
    doneReentered();
  }

  /// Take every published entry, appending those stamped \p seq to \p out in
  /// oldest-first order and dropping the rest.
  /// \pre the caller is the ring's only consumer, serialised by SharedGC's
  ///   barrier mutex, and \p seq was read under that same mutex.
  /// \pre a collector calling this reads slots with acquire loads. Once it
  ///   has read a value a barrier stored with a release, that is what
  ///   guarantees the tail read here covers the push made before that store.
  void drainInto(std::vector<SharedGCCell *> &out, uint32_t seq);

  /// Take every published entry without reading it -- the only operation that
  /// is safe on residue whose cells an earlier cycle may already have swept.
  /// \pre the caller is the ring's only consumer.
  void discardAll() {
    head_.store(
        tail_.load(std::memory_order_acquire), std::memory_order_release);
  }

  /// \return the number of published entries not yet consumed. Test-only.
  uint32_t sizeSlow() const {
    return tail_.load(std::memory_order_acquire) -
        head_.load(std::memory_order_acquire);
  }

 private:
  Entry entries_[kCapacity];

  /// Monotone and masked on use, so "full" is exactly tail_ - head_ ==
  /// kCapacity. Written only by the producer.
  std::atomic<uint32_t> tail_{0};
  /// Written only by a consumer. Release, so that consumer's reads of entries_
  /// are ordered ahead of the producer reusing those slots.
  std::atomic<uint32_t> head_{0};
  /// The producer's cache of head_, stale only in the safe direction: it
  /// over-reports occupancy, costing an early spill rather than an overwrite.
  uint32_t headCache_{0};

#ifndef NDEBUG
  /// Set for the duration of a push, to catch two producers at once. A latched
  /// thread id would instead reject an embedder legitimately driving one
  /// Runtime from several threads, which Hermes permits.
  std::atomic<bool> pushing_{false};

  bool reentered() {
    return pushing_.exchange(true, std::memory_order_relaxed);
  }
  void doneReentered() {
    pushing_.store(false, std::memory_order_relaxed);
  }
#else
  bool reentered() {
    return false;
  }
  void doneReentered() {}
#endif
};

} // namespace vm
} // namespace hermes
