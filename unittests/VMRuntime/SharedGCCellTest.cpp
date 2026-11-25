/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/CellKind.h"
#include "hermes/VM/SharedHeap/SharedGCCell.h"

namespace {

using namespace hermes;
using namespace hermes::vm;

/// Every shared kind, so the checks below extend automatically.
constexpr CellKind kSharedKinds[] = {
    CellKind::SharedArrayKind,
};

TEST(SharedGCCellTest, Kind_RoundTrips) {
  // Raw storage, as SharedGC hands out: no constructor runs.
  alignas(SharedGCCell) char buf[sizeof(SharedGCCell)];
  auto *cell = reinterpret_cast<SharedGCCell *>(buf);

  cell->setKind(CellKind::SharedArrayKind);
  EXPECT_EQ(CellKind::SharedArrayKind, cell->getKind());
}

TEST(SharedGCCellTest, IsSharedCellKind_AcceptsOnlySharedKinds) {
  for (CellKind kind : kSharedKinds)
    EXPECT_TRUE(isSharedCellKind(kind)) << cellKindStr(kind);

  EXPECT_FALSE(isSharedCellKind(CellKind::JSObjectKind));
  EXPECT_FALSE(isSharedCellKind(CellKind::UninitializedKind));
  EXPECT_FALSE(isSharedCellKind(CellKind::NativeJSFunctionKind));
}

TEST(SharedGCCellTest, SharedKinds_AreOutsideTheObjectRange) {
  // A shared cell is not a JSObject: JS reaches one only through a
  // wrapper. If these kinds ever drift inside the Object range,
  // vmcast<JSObject> would succeed on a cell that has no JSObject layout.
  for (CellKind kind : kSharedKinds) {
    EXPECT_FALSE(kindInRange(
        kind, CellKind::ObjectKind_first, CellKind::ObjectKind_last))
        << cellKindStr(kind);
    // Not part of AllCells.
    EXPECT_FALSE(kindInRange(
        kind, CellKind::AllCellsKind_first, CellKind::AllCellsKind_last))
        << cellKindStr(kind);
  }
}

} // namespace
