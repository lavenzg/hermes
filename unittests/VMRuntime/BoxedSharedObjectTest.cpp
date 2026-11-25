/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/VM/BoxedSharedObject.h"

#include "VMRuntimeTestHelpers.h"
#include "gtest/gtest.h"

using namespace hermes::vm;

namespace {

using BoxedSharedObjectTest = RuntimeTestFixture;

/// A fabricated address, never dereferenced. Nothing in this diff reads through
/// a box's cell -- the interception that would is not written yet -- so a real
/// shared allocation would only obscure what is under test.
static SharedGCCell *fakeCell(uintptr_t n) {
  return reinterpret_cast<SharedGCCell *>(n);
}

TEST_F(BoxedSharedObjectTest, Create_IsARealObjectOfItsOwnKind) {
  GCScope scope{runtime, "BoxedSharedObjectTest"};
  auto box = runtime.makeHandle(BoxedSharedObject::create(runtime));

  EXPECT_EQ(CellKind::BoxedSharedObjectKind, box->getKind());
  EXPECT_TRUE(vmisa<BoxedSharedObject>(box.get()));
  // It has to pass as a JSObject: anything Object-tagged that is not one is a
  // wild read rather than a clean miss.
  EXPECT_TRUE(vmisa<JSObject>(box.get()));
}

TEST_F(BoxedSharedObjectTest, Create_HasANullParent) {
  // What the shared struct spec requires, and what keeps a box clear of the
  // negative-match cache.
  GCScope scope{runtime, "BoxedSharedObjectTest"};
  auto box = runtime.makeHandle(BoxedSharedObject::create(runtime));
  EXPECT_EQ(nullptr, box->getParent(runtime));
}

TEST_F(BoxedSharedObjectTest, Create_DoesNotShareOrdinaryObjectsClass) {
  // A box's class comes from a root of its own, so a box never shares a class
  // with an ordinary object.
  GCScope scope{runtime, "BoxedSharedObjectTest"};
  auto nullObj = runtime.makeNullHandle<JSObject>();
  auto ordinary = runtime.makeHandle(JSObject::create(runtime, nullObj));
  auto box = runtime.makeHandle(BoxedSharedObject::create(runtime));
  EXPECT_NE(ordinary->getClass(runtime), box->getClass(runtime));
}

TEST_F(BoxedSharedObjectTest, Create_ReservesTheOverlapSlots) {
  // The slots that the box's own field overlaps are reserved in its class, so
  // no property can be stored over the cell pointer.
  struct : public Locals {
    PinnedValue<BoxedSharedObject> box;
  } lv;
  LocalsRAII lraii{runtime, &lv};
  lv.box = BoxedSharedObject::create(runtime);
  EXPECT_EQ(
      JSObject::numOverlapSlots<BoxedSharedObject>(),
      lv.box->getClass(runtime)->getNumProperties());
}

TEST_F(BoxedSharedObjectTest, Create_CannotGainAPrototype) {
  // Not extensible, so the parent stays null: a box given a prototype would
  // share a class with a non-null parent, which the negative-match cache reads.
  struct : public Locals {
    PinnedValue<BoxedSharedObject> box;
    PinnedValue<JSObject> proto;
  } lv;
  LocalsRAII lraii{runtime, &lv};
  lv.box = BoxedSharedObject::create(runtime);
  lv.proto = JSObject::create(runtime);
  EXPECT_FALSE(lv.box->isExtensible(runtime));
  auto res = JSObject::setParent(*lv.box, runtime, *lv.proto);
  ASSERT_NE(ExecutionStatus::EXCEPTION, res.getStatus());
  EXPECT_FALSE(*res);
  EXPECT_EQ(nullptr, lv.box->getParent(runtime));
}

TEST_F(BoxedSharedObjectTest, Bind_HoldsTheCellItWasGiven) {
  GCScope scope{runtime, "BoxedSharedObjectTest"};
  auto *cell = fakeCell(0x1000);
  auto box = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
  EXPECT_EQ(cell, box->getSharedCell());
}

TEST_F(BoxedSharedObjectTest, Bind_CanHappenAfterConstruction) {
  // The two-phase form the allocation protocol needs: a box is created before
  // the shared cell exists, so that binding and publishing can happen with no
  // allocation in between.
  GCScope scope{runtime, "BoxedSharedObjectTest"};
  auto box = runtime.makeHandle(BoxedSharedObject::create(runtime));
  ASSERT_EQ(nullptr, box->getSharedCell());

  auto *cell = fakeCell(0x2000);
  box->setSharedCell(cell);
  EXPECT_EQ(cell, box->getSharedCell());
}

TEST_F(BoxedSharedObjectTest, Unbound_SurvivesALocalGC) {
  // The allocation protocol rests on this: it creates the box before the
  // cell, so a box with no cell bound must survive a local GC.
  GCScope scope{runtime, "BoxedSharedObjectTest"};
  auto box = runtime.makeHandle(BoxedSharedObject::create(runtime));
  ASSERT_EQ(nullptr, box->getSharedCell());

  runtime.collect("test");

  EXPECT_EQ(nullptr, box->getSharedCell());
  EXPECT_TRUE(vmisa<BoxedSharedObject>(box.get()));
  // And it is still usable afterwards.
  auto *cell = fakeCell(0x3000);
  box->setSharedCell(cell);
  EXPECT_EQ(cell, box->getSharedCell());
}

TEST_F(BoxedSharedObjectTest, Bound_SurvivesALocalGCWithItsCellIntact) {
  // The cell is a raw pointer into a heap the local collector knows nothing
  // about. If the overlap-slot count in BuildMeta were wrong, the collector
  // would decode these bytes as a GCSmallHermesValue. This made-up address
  // decodes as a non-pointer in every mode, so that would go unnoticed here.
  GCScope scope{runtime, "BoxedSharedObjectTest"};
  auto *cell = fakeCell(0x4000);
  auto box = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));

  runtime.collect("test");
  runtime.collect("test");

  EXPECT_EQ(cell, box->getSharedCell());
}

TEST_F(BoxedSharedObjectTest, Boxes_AreDistinctObjectsPerCell) {
  // create() does no deduplication; the runtime's map will. This pins that the
  // dedup invariant is not accidentally satisfied here, so a later diff cannot
  // quietly rely on create() for it.
  GCScope scope{runtime, "BoxedSharedObjectTest"};
  auto *cell = fakeCell(0x5000);
  auto first = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));
  auto second = runtime.makeHandle(BoxedSharedObject::create(runtime, cell));

  EXPECT_NE(first.get(), second.get());
  EXPECT_EQ(cell, first->getSharedCell());
  EXPECT_EQ(cell, second->getSharedCell());
}

} // namespace
