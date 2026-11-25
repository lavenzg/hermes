/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/VM/CellKind.h"
#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGCCell.h"
#include "hermes/VM/SharedHeap/SharedValue.h"

#include "llvh/Support/Casting.h"
#include "llvh/Support/ErrorHandling.h"

namespace hermes {
namespace vm {

/// Enumerates the slots of a shared cell and calls an acceptor on each.
///
/// Callers go through SharedGC::markCell rather than calling visit() directly,
/// mirroring the way GCBase::markCell wraps SlotVisitor::visit.
namespace SharedSlotVisitor {
namespace detail {

/// One overload per shared cell kind, selected by the switch in visit(). A new
/// SHARED_CELL_KIND with no overload here fails to compile: the switch is
/// generated from CellKinds.def, so the kind emits a call that binds to
/// nothing. The call is qualified, so every overload must be declared above
/// visit().
template <typename Acceptor>
void visitSlots(Acceptor &acceptor, SharedArray *cell) {
  // length(), never SharedGC::allocatedSize(): the size bucket rounds up and
  // slack past end() was never initialised.
  // One atomic load per slot, which is what getPointerOrNull is for: this is
  // the collector reading a slot while the owning runtime stores into it, and
  // testing then decoding would be two reads of a value that can change
  // between them.
  for (SharedValue *p = cell->begin(), *e = cell->end(); p != e; ++p)
    if (SharedGCCell *c = p->getPointerOrNull())
      acceptor.accept(c);
}

} // namespace detail

/// Call \p acceptor on every slot of \p cell that holds a pointer, in address
/// order.
///
/// Decoding is the visitor's job, not each acceptor's: an empty slot is
/// skipped here.
/// Note: we expect very few cell kinds in the shared heap, so a simple switch
/// should be efficient enough.
template <typename Acceptor>
void visit(Acceptor &acceptor, SharedGCCell *cell) {
  switch (cell->getKind()) {
#define SHARED_CELL_KIND(name) \
  case CellKind::name##Kind:   \
    return detail::visitSlots(acceptor, llvh::cast<name>(cell));
#include "hermes/VM/CellKinds.def"
    default:
      llvm_unreachable("not a shared cell kind");
  }
}

} // namespace SharedSlotVisitor
} // namespace vm
} // namespace hermes
