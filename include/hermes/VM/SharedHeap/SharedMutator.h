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
#include <cassert>
#include <cstdint>

namespace hermes {
namespace vm {

class Runtime;

/// One Runtime's barrier state: its ring, its in-flight counter, and its
/// registration with the group.
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
  /// Unregisters, which reads the runtime's box map and takes its lock, so the
  /// runtime must still be alive.
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
  /// With marking off, this is one seq_cst load and a predictable branch. The
  /// load must be seq_cst, not acquire, for two reasons. The pre-read below
  /// relies on the total order; see SharedValue::set. So does the in-flight
  /// check (see enterBarrier): on AArch64 an acquire load may become an ldapr,
  /// which can pass the entry's stlr.
  ///
  /// **Run this inside a barrier window that also covers the store**, as
  /// SharedValue::set does. Reading the flag is not a handshake: a thread
  /// preempted between this load and its store would land the store after
  /// marking turned on, unrecorded. The window is what makes the collector wait
  /// for that store before it traces.
  ///
  /// With marking on, the pre-read and the store that follows it in
  /// SharedValue::set are both seq_cst; the comment on that store says why.
  void snapshotWriteBarrier(const SharedValue &slot) {
    assert(barrierDepth_ != 0 && "the store barrier ran outside a window");
    // One load says whether marking is on and gives the stamp. Two atomics
    // could not: marking could turn on between the two loads.
    const uint32_t seq = gc_.cycleSeq();
    if (LLVM_LIKELY(seq % 2 == 0))
      return;
    SharedGCCell *old = slot.getPointerOrNullSeqCst();
    if (old)
      push(old, seq);
  }

  /// Open a barrier window, which runs from a test of the marking flag through
  /// the effect that test guards. Before it traces, the collector waits for
  /// every window that was open when marking turned on, and then for every
  /// window open when beginSecondInFlightPass looks.
  ///
  /// Windows nest, and only the outermost entry and exit write the counter.
  /// The entry store is seq_cst, and so are the window's load of cycleSeq_,
  /// the collector's increment of it, and the collector's read of this
  /// counter. In that total order, either the window sees marking on, or the
  /// collector's read comes after this entry: it then finds the window open
  /// and waits, or finds it closed with its effect already visible. With
  /// anything weaker both could miss.
  ///
  /// \pre the caller owns the Runtime this belongs to.
  void enterBarrier() {
    if (barrierDepth_++ != 0)
      return;
    const uint32_t count = inFlight_.load(std::memory_order_relaxed);
    assert(count % 2 == 0 && "the in-flight counter is out of step");
    inFlight_.store(count + 1, std::memory_order_seq_cst);
  }

  /// Close the window the matching enterBarrier opened. Release is enough:
  /// this store only has to carry the window's effect -- the slot store, for
  /// the store barrier -- to a collector that sees the counter move, and any
  /// later counter store the collector might read instead releases too.
  ///
  /// \pre the caller owns the Runtime this belongs to.
  void exitBarrier() {
    assert(barrierDepth_ != 0 && "exitBarrier without a matching enterBarrier");
    if (--barrierDepth_ != 0)
      return;
    const uint32_t count = inFlight_.load(std::memory_order_relaxed);
    assert(count % 2 == 1 && "the in-flight counter is out of step");
    inFlight_.store(count + 1, std::memory_order_release);
  }

#ifdef UNIT_TEST
  /// \return entries published to the ring and not yet consumed.
  uint32_t ringSizeSlow() const {
    return ring_.sizeSlow();
  }

  /// \return the in-flight counter, odd while a window is open.
  uint32_t inFlightCounterSlow() const {
    return inFlight_.load(std::memory_order_relaxed);
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

  /// Odd while this runtime is inside a barrier window. Only this runtime
  /// writes it; the collector reads it after turning marking on. Wrapping is
  /// harmless: the collector waits for any change from an odd value, and that
  /// value changes only when its window closes.
  std::atomic<uint32_t> inFlight_{0};

  /// How many windows are open. Only this runtime touches it.
  uint32_t barrierDepth_{0};

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
  // The window covers the store, not just the barrier: with marking off the
  // store is the barrier's whole effect, and it is what the collector waits
  // for.
  mutator.enterBarrier();
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
  // marking off: a prune, or a detach. A barrier in another runtime that sees
  // marking on must still pre-read this value: if it read an older one, it
  // would record that, overwrite this one unrecorded, and the cell could be
  // freed while some runtime still reaches it. Release and acquire alone allow
  // that stale read; the total order rules it out.
  raw_.store(v.raw_.load(std::memory_order_relaxed), std::memory_order_seq_cst);
  mutator.exitBarrier();
}

} // namespace vm
} // namespace hermes
