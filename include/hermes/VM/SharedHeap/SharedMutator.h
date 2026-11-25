/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/Support/Compiler.h"
#include "hermes/VM/SharedHeap/SharedBarrierRing.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedValue.h"

#include <atomic>
#include <cstdint>

namespace hermes {
namespace vm {

class Runtime;

/// One Runtime's barrier state: its ring, and its registration with the group.
///
/// A Runtime owns exactly one of these and holds it for its whole life, so the
/// collector can reach every runtime's ring by walking the registry. Splitting
/// it out of SharedGC is what keeps the barrier's fast path off any shared
/// line: a store appends to this object's own ring and touches nothing another
/// thread writes.
class SharedMutator {
 public:
  /// Registers with \p gc as \p runtime's entry.
  ///
  /// \p runtime is held by reference for the mutator's whole life and is how
  /// the collector reaches that runtime's box map, which is its root set. It
  /// is not dereferenced here, and must outlive this.
  SharedMutator(SharedGC &gc, Runtime &runtime);
  /// Unregisters, which takes the runtime's map lock, so the runtime must still
  /// be alive.
  ~SharedMutator();

  SharedMutator(const SharedMutator &) = delete;
  SharedMutator &operator=(const SharedMutator &) = delete;

  /// Append \p cell, stamped \p seq, spilling the ring into the barrier
  /// worklist first if it is full.
  ///
  /// Safe with no cycle running: a barrier preempted across the end of one
  /// arrives here, and the spill then discards the ring rather than reading
  /// cells that may already be swept.
  /// \pre the caller owns the Runtime this belongs to, and \p seq is odd --
  ///   that is, it came from a cycleSeq() read that found marking on.
  void push(SharedGCCell *cell, uint32_t seq);

  /// The SATB barrier: record whatever \p slot holds, before it is
  /// overwritten.
  ///
  /// With marking off, this is one seq_cst load and a predictable branch.
  /// seq_cst rather than acquire because the pre-read below relies on the
  /// total order; see SharedValue::set.
  ///
  /// **Reading the flag is not a handshake, and this alone is not yet sound.**
  /// Nothing sequences the *store* this guards against turning marking on --
  /// only the load. A thread preempted between the two commits a store that
  /// violates the snapshot and records nothing. So the collector has to wait,
  /// after turning marking on, until no runtime is inside a barrier; until it
  /// does, see beginCycle: a cycle may only start when no mutator is running.
  ///
  /// With marking on, the pre-read and the store that follows it in
  /// SharedValue::set are both seq_cst; the comment on that store says why.
  void snapshotWriteBarrier(const SharedValue &slot) {
    // One load says whether marking is on and gives the stamp. Two atomics
    // could not: marking could turn on between the two loads.
    const uint32_t seq = gc_.cycleSeq();
    if (LLVM_LIKELY(seq % 2 == 0))
      return;
    SharedGCCell *old = slot.getPointerOrNullSeqCst();
    if (old)
      push(old, seq);
  }

#ifdef UNIT_TEST
  /// \return entries published to the ring and not yet consumed.
  uint32_t ringSizeSlow() const {
    return ring_.sizeSlow();
  }
#endif

  /// The runtime this belongs to.
  Runtime &runtime() const {
    return runtime_;
  }

 private:
  friend class SharedGC;

  SharedGC &gc_;
  Runtime &runtime_;
  SharedBarrierRing ring_;

#ifndef NDEBUG
  /// Set before the destructor unregisters, and asserted by push.
  ///
  /// The ring's own guard cannot cover this: it catches two concurrent pushes,
  /// deliberately not a latched thread id, so it never sees a push racing the
  /// destructor. Without this, that race writes into freed Runtime storage
  /// with no diagnostic at all.
  bool detached_{false};
#endif
};

inline void SharedValue::set(SharedValue v, SharedMutator &mutator) {
  mutator.snapshotWriteBarrier(*this);
  // seq_cst, which does two jobs.
  //
  // As a release, it makes the barrier's push visible to whoever reads the new
  // value with acquire, so the collector never sees the new value without the
  // record of the old one. A newly built cell's construction is published the
  // same way.
  //
  // It also puts this store in one total order with every test of the marking
  // flag and with the barrier's seq_cst pre-read. Suppose this runtime then
  // gives up its own key for the cell it stores here, after a test that saw
  // marking off. A barrier in another runtime that sees marking on must still
  // pre-read this value: if it read an older one, it would record that,
  // overwrite this one unrecorded, and the cell could be freed while some
  // runtime still reaches it. Release and acquire alone allow that stale read;
  // the total order rules it out.
  raw_.store(v.raw_.load(std::memory_order_relaxed), std::memory_order_seq_cst);
}

} // namespace vm
} // namespace hermes
