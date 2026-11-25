/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/VM/SharedHeap/SharedBarrierRing.h"

namespace hermes {
namespace vm {

void SharedBarrierRing::drainInto(
    std::vector<SharedGCCell *> &out,
    uint32_t seq) {
  // Pairs with the producer's release store of tail_, and is the only
  // synchronisation the consumer has with it.
  const uint32_t t = tail_.load(std::memory_order_acquire);
  // Relaxed: head_ has no other writer.
  for (uint32_t h = head_.load(std::memory_order_relaxed); h != t; ++h) {
    const Entry &e = entries_[h & (kCapacity - 1)];
    // An older stamp may name a cell that sweep has freed and releaseRegion
    // has unmapped. Nothing dereferences e.cell before this test.
    if (e.seq == seq)
      out.push_back(e.cell);
  }
  // Orders the reads above ahead of the producer reusing those slots.
  head_.store(t, std::memory_order_release);
}

} // namespace vm
} // namespace hermes
