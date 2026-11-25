/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedSlotVisitor.h"

namespace hermes {
namespace vm {

template <typename Acceptor>
inline void SharedGC::markCell(Acceptor &acceptor, SharedGCCell *cell) {
  SharedSlotVisitor::visit(acceptor, cell);
}

} // namespace vm
} // namespace hermes
