/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/VM/BoxedSharedObject.h"

#include "hermes/VM/BuildMetadata.h"
#include "hermes/VM/Runtime-inline.h"

namespace hermes {
namespace vm {

void BoxedSharedObjectBuildMeta(const GCCell *cell, Metadata::Builder &mb) {
  // The local GC must skip sharedCell_, or it would decode the shared address
  // as a GCSmallHermesValue.
  constexpr unsigned kOverlapSlots =
      JSObject::numOverlapSlots<BoxedSharedObject>();
  static_assert(
      offsetof(BoxedSharedObject, sharedCell_) + sizeof(SharedGCCell *) <=
          offsetof(JSObjectAndDirectProps, directProps_) +
              kOverlapSlots * sizeof(GCSmallHermesValue),
      "the overlap slots must cover sharedCell_");
  mb.addJSObjectOverlapSlots(kOverlapSlots);
  JSObjectBuildMeta(cell, mb);
  mb.setVTable(&BoxedSharedObject::vt);
}

// TODO: we could represent the edge to the shared heap in the heap snapshot,
// probably as a native memory edge.
const ObjectVTable BoxedSharedObject::vt{
    VTable(CellKind::BoxedSharedObjectKind, cellSize<BoxedSharedObject>()),
    BoxedSharedObject::_getOwnIndexedRangeImpl,
    BoxedSharedObject::_haveOwnIndexedImpl,
    BoxedSharedObject::_getOwnIndexedPropertyFlagsImpl,
    BoxedSharedObject::_getOwnIndexedImpl,
    BoxedSharedObject::_setOwnIndexedImpl,
    BoxedSharedObject::_deleteOwnIndexedImpl,
    BoxedSharedObject::_checkAllOwnIndexedImpl,
};

PseudoHandle<BoxedSharedObject> BoxedSharedObject::createImpl(
    Runtime &runtime,
    SharedGCCell *cell) {
  auto *box = runtime.makeAFixed<BoxedSharedObject>(
      runtime, runtime.boxedSharedObjectClass, cell);
  return JSObjectInit::initToPseudoHandle(runtime, box);
}

} // namespace vm
} // namespace hermes
