/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/VM/JSObject.h"
#include "hermes/VM/SharedHeap/SharedGCCell.h"

namespace hermes {
namespace vm {

/// One runtime's handle on one cell in the shared heap.
///
/// **Opaque, not transparent.** Unlike BoxedDouble there is no automatic
/// unboxing; the interpreter treats a box as an ordinary value, and will check
/// for this special kind in slow path to handle shared heap access.
class BoxedSharedObject final : public JSObject {
  friend void BoxedSharedObjectBuildMeta(const GCCell *, Metadata::Builder &);

 public:
  static const ObjectVTable vt;

  static constexpr CellKind getCellKind() {
    return CellKind::BoxedSharedObjectKind;
  }
  static bool classof(const GCCell *cell) {
    return cell->getKind() == CellKind::BoxedSharedObjectKind;
  }

  /// Create a box bound to no cell yet. Whoever creates it binds it with
  /// setSharedCell and records it in the runtime's map.
  /// The parent is null, and a box is not extensible, so it stays null. That
  /// is what the shared struct spec requires anyway, and it is also what keeps
  /// a box clear of the negative-match cache.
  static PseudoHandle<BoxedSharedObject> create(Runtime &runtime) {
    return createImpl(runtime, nullptr);
  }

#ifdef UNIT_TEST
  /// Create a box already bound to \p cell. Test-only: the box is in no map,
  /// so it does not make its cell a root.
  static PseudoHandle<BoxedSharedObject> create(
      Runtime &runtime,
      SharedGCCell *cell) {
    return createImpl(runtime, cell);
  }
#endif

  /// \return the shared cell this box stands for, or null if it is not bound
  /// yet.
  SharedGCCell *getSharedCell() const {
    return sharedCell_;
  }

  /// Bind this box to \p cell, once.
  /// \pre sharedCell_ is nullptr. So a valid boxed object can't be overwritten
  /// to hold another shared cell.
  void setSharedCell(SharedGCCell *cell) {
    assert(cell && "cannot bind a box to nothing");
    assert(!sharedCell_ && "a box is bound once and never retargeted");
    sharedCell_ = cell;
  }

  BoxedSharedObject(
      Runtime &runtime,
      Handle<HiddenClass> clazz,
      SharedGCCell *cell)
      : JSObject(runtime, nullptr, *clazz), sharedCell_(cell) {}

 private:
  /// The body of both create overloads.
  static PseudoHandle<BoxedSharedObject> createImpl(
      Runtime &runtime,
      SharedGCCell *cell);

  /// Raw pointer to the cell in the shared heap. Local GC will not trace this.
  /// Instead, it's provided as roots to the collector of the shared heap. It
  /// can never be invalid as long as this boxed object is alive.
  SharedGCCell *sharedCell_;
};

} // namespace vm
} // namespace hermes
