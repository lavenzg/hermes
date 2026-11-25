/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/Support/Compiler.h"
#include "hermes/VM/SharedHeap/SharedBarrierRing.h"

#include <cstdint>

namespace hermes {
namespace vm {

class Runtime;
class SharedGC;

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
  /// \pre the caller owns the Runtime this belongs to.
  void push(SharedGCCell *cell, uint32_t seq);

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

} // namespace vm
} // namespace hermes
