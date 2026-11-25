# A shape-driven object model for the shared heap

Date: 2026-08-26
Revised: 2026-10-21
Status: **Proposed. Not implemented, and deliberately deferred.**
Relates to: [allocator.md](../../../plans/shared-heap/allocator.md) (size buckets),
[boxing.md](../../../plans/shared-heap/boxing.md) (boxes, field lookup),
[marking.md](../../../plans/shared-heap/marking.md) (black allocation, P4, P6, P8),
[barrier.md](../../../plans/shared-heap/barrier.md) (store barrier),
[marking-implementation.md](../../../plans/shared-heap/marking-implementation.md) (the tracer this
would replace)

---

## Summary

This doc covers how a shared cell records its type and how the collector finds its slots. It
proposes replacing the one-byte `CellKind` at the start of every shared cell with a pointer to a
*shape*, a shared cell that describes the type. The tracer becomes one loop with no per-kind
`switch`, and shared structs get an 8-byte header. In exchange, array headers grow from 8 to 16
bytes on 64-bit targets.

It is deferred because the trade depends on whether struct or array instances dominate, and
`SharedStruct` is not designed. Adopting it now would charge `SharedArray`, the only shared cell
type, for uniformity with a type that does not exist. Adopting it later is additive (struct shapes
gain the fields the tracer reads, arrays gain a shape pointer, the `switch` becomes a loop);
undoing it would not be.

**What exists today (implemented).** In release builds `SharedGCCell` holds only a one-byte
`CellKind`, and `SharedArray` keeps `uint32_t length_` in the padding after it. Each slot is a
`SharedValue`: one 8-byte `std::atomic<uint64_t>` holding a shared-cell pointer or nothing.
`SharedSlotVisitor::visit` switches on the kind (one case) and walks `length()` slots. Shared kinds
have no `VTable` or `Metadata`: their `SHARED_CELL_KIND` entries come after every local kind, past
the end of the kind-indexed tables.

## The proposal (not implemented)

```cpp
class SharedGCCell {
  SharedShape *shape_;            // 8 bytes; the kind lives in the shape
};

class SharedShape final : public SharedGCCell {
 public:
  /// Every slot is one SharedValue wide: a constant, not a per-shape field.
  static constexpr uint32_t kSlotStride = sizeof(SharedValue);
  /// Where a variable-length cell keeps its uint32_t count: after the base.
  static constexpr uint16_t kLengthOffset = sizeof(SharedGCCell);
  /// slotCount == kInlineLength means "read the count at kLengthOffset".
  static constexpr uint32_t kInlineLength = UINT32_MAX;

  uint32_t slotCount;
  uint16_t slotsOffset;
  CellKind kind;
};

class SharedArray final : public SharedGCCell {
  uint32_t length_;               // at kLengthOffset; JS-observable as `length`
  // 4 bytes of padding, free for flags later
  // SharedValue elements[length_] at offset 16 in release builds
};

template <typename Acceptor>
void traceSharedCell(Acceptor &acc, SharedGCCell *cell) {
  SharedShape *sh = cell->shape();
  acc.accept(sh);                       // the shape edge keeps the type alive
  char *base = reinterpret_cast<char *>(cell);
  const uint32_t n = sh->slotCount == SharedShape::kInlineLength
      ? *reinterpret_cast<uint32_t *>(base + SharedShape::kLengthOffset)
      : sh->slotCount;
  char *p = base + sh->slotsOffset;
  for (uint32_t i = 0; i < n; ++i, p += SharedShape::kSlotStride)
    if (SharedGCCell *c = reinterpret_cast<SharedValue *>(p)->getPointerOrNull())
      acc.accept(c);
}
```

The tracer needs no `CellKind`, and a new cell kind needs no change to it. A constant stride lets
the compiler fold the loop's increment; a stride field is needed only if some cell's pointer slots
are not 8 bytes wide. The loop gives up the kind switch's compile-time guard
(marking-implementation.md, "Finding a cell's slots"): a shape whose `slotCount` or `slotsOffset` is
wrong skips slots or follows bytes that are not slots, so each built-in shape's fields need a
`static_assert` against its class's layout, like the one for `kLengthOffset` below.

### Why the count has two representations

A struct's field count belongs to its type, so it lives in the shape. An array's length belongs to
the instance, so it lives in the cell. Both are fixed at construction (TC39 fixes a `SharedArray`'s
length; a shape never changes once built), so the tracer reads the count and `slotsOffset` with
plain loads; only the slots are atomic. Publication ordering (below) still applies.

The length is not derived from the cell's size. `length` is observable from JS, so it must be
stored exactly anyway, and the only size available, the cell's size bucket
(`SharedGC::allocatedSize`), is rounded up. Buckets step by one 8-byte slot up to 128 bytes, then
come four per doubling up to 8 KiB: a 200-byte array lands in the 224-byte bucket, and a derived
count would follow three unwritten slots as pointers. Above 8 KiB, `allocatedSize` is the cell's
whole region. Zeroing the slack would lengthen the constructor's `memset`, which runs under `mtx_`,
by up to a fifth of a bucketed cell, or most of a 64 KiB segment for a cell just over 8 KiB.

One field, `slotCount`, says whether the count is in the cell and, if not, what it is. That works
because an in-cell count is always at `kLengthOffset`. Assert that inside `SharedArray` (in its
constructor, for example), since `length_` is private:
`static_assert(offsetof(SharedArray, length_) == SharedShape::kLengthOffset)`.
Hermes builds with `-Wno-invalid-offsetof`, so `offsetof` works although the base class has data.
The sentinel is `UINT32_MAX`, not 0, because `shared struct Empty {}` is a legal type with zero
slots.

### Keeping shapes alive

Every cell depends on its shape: a dangling `shape_` breaks tracing and every kind test. The
`HERMES_SLOW_DEBUG` re-trace after each certified cycle checks a cell against the region headers
before it reads the cell, so it must check `shape_` the same way before it reads the shape's fields.

* **Meta-shape.** As in V8, one meta-shape is its own shape. Here it has `slotCount = 0`, because
  shapes have no traced fields: tracing a shape visits only its shape edge, and the meta-shape's
  edge to itself stops at the already-marked check.
* **No barrier on `shape_`.** It is written once, under `mtx_` while `makeALocked` builds the cell,
  before any other thread can reach it. As for `SharedValue::initField`, that is the only valid
  reason to skip the barrier. "Nothing was overwritten" is not one: `set`'s seq_cst store matters
  even when the barrier records nothing (marking.md P5). A barriered store would also invert the
  lock order: a barrier with a full ring takes `barrierMtx_`, which must come before `mtx_`.
* **Struct shapes: held through boxes.** A runtime that creates a struct must hold its shape through
  a box (a `BoxedSharedObject`, the JS object through which a runtime refers to a shared cell), as
  for every shared cell it references; box-map keys are the collector's roots. New cells are born
  marked ("black allocation") and are not traced in the cycle they are born in, so a new cell's
  `shape_` edge is covered like its other edges: by P4-dominance (marking.md P8), which needs the
  acquisition push (P4), the prune push (P6) and the copy of a leaving runtime's keys (P12).
* **Built-in shapes: permanent roots** (open). `SharedGC` holds the meta-shape and the shape all
  arrays share directly. No runtime acquires them, and `SharedGC`'s pointers are not roots, so a
  cycle sweeps a built-in shape whenever no traced cell points at it: for example, when no array is
  live, or every live array was created during the cycle. Later cells would get a dangling
  `shape_`. Built-in shapes must be permanent roots or never swept. Either way `~SharedGC` must drop
  them before its last collection, which assumes nothing is a root and then asserts that the heap is
  empty (boxing.md O-B4).

## Measured layouts

All figures come from compiling the candidate layouts (release, 64-bit). Slots are 8 bytes and must
be 8-aligned, which drives every number below.

| | array header | struct header |
|---|---|---|
| **Current base:** kind in `SharedGCCell`; array keeps `length_` inline; a struct would keep its shape pointer in its own class (hypothetical) | **8** | 16 |
| **Proposed:** shape pointer in `SharedGCCell`; the shape holds the count or a sentinel | 16 | **8** |
| Shape pointer in `SharedGCCell`; count always inline | 16 | 16 |
| 32-bit shape *index* in `SharedGCCell` | 8 | 8 |

A four-element `SharedArray` is 48 bytes under the proposal and 40 today. `SharedShape` is 16 bytes
with the fields above, or 24 with separate `lengthOffset` and `slotStride` fields. On 32-bit
targets a pointer is 4 bytes, so every row is 8 and 8 (by arithmetic, not measured).

Per instance, the current base and the proposal tie: the proposal moves 8 bytes from structs to
arrays, a uniformity win rather than a size win. It also makes `getKind()` a dependent load
(`shape_->kind`): a second cache line, though a hot one, on paths that only need the kind.

## Rejected alternatives

* **A 32-bit shape index instead of a pointer.** Smallest on 64-bit (8 bytes for both; 16 bits
  would do if shapes are few). But shape lifetime would move from a GC edge into a table that pins
  every shape forever or needs its own sweeping. Reconsider if header bytes come to matter more, or
  if struct types prove bounded so shapes never need freeing. boxing.md's field lookup already
  needs a shared shape registry (O-B13), which faces the same lifetime choice.
* **The length in the cell's tail slack**, to keep an 8-byte header. The slack can be zero, so the
  allocation would have to reserve 8 bytes anyway.
* **Reusing `Metadata` and `SlotVisitor`.** `Metadata::ArrayData`'s offsets and stride fit, but its
  `type` comes from `SlotKinds.def`, so a shared entry would widen an enum the local `SlotVisitor`
  switches over. `Metadata::offset_t` is `uint8_t` (offsets up to 255), and `VTable` is typed on
  `GCCell` and `GC`.
* **Always storing the count inline, with no sentinel.** Uniform, but a two-field struct grows from
  24 to 32 bytes to save one compare-and-select per traced cell; `slotCount` sits on the shape line
  the tracer already reads for `slotsOffset`.

## Open questions

1. **How non-pointer values are encoded.** boxing.md plans to store `undefined`, `null`, booleans
   and numbers inline in the slot; `SharedValue` has no encoding for them yet, and an empty slot is
   meant to read as `undefined`. The encoding must fit in 8 bytes, as every layout figure assumes,
   and keep the tracer at one load per slot: `getPointerOrNull` should return null for a
   non-pointer, because testing `isPointer()` and then decoding is two loads of a slot another
   runtime may be writing. Until this is settled `SharedStruct` cannot hold non-pointer fields,
   which blocks this design.
2. **Can struct types be created without bound?** If each struct type comes from a declaration
   evaluated once, or types are deduplicated (for example by a registry), shapes are bounded by
   program text and could simply never be freed. If a declaration can be evaluated repeatedly, as
   in a loop, shapes need real lifetimes, which argues for a traced shape.
3. **Do structs or arrays dominate?** This decides the layout trade.
4. **Where the field-name table lives** once property access exists. Uniform slots mean a
   name-to-index map suffices; an offsets array is needed only if slot widths vary. A shape cannot
   own malloc'd memory: no shared cell is ever destroyed (`makeALocked` static-asserts trivial
   destructibility). It cannot allocate the table either: its constructor runs under `mtx_`. If the
   table is a separate shared cell, shapes gain a traced slot and the meta-shape's `slotCount` is no
   longer 0. Field lookup also needs a per-runtime symbolization table (boxing.md O-B13). Neither
   table is designed.
5. **Publication ordering** (boxing.md O-B10). For slots it is implemented: `set()` stores with
   seq_cst, which is a release, and every slot read that may be followed to a cell is an acquire,
   the tracer's included, so the tracer sees `kind_` and `length_` as the cell was built. The
   proposal adds `shape_` and the shape's fields, which are ordered only if every way of reaching a
   shape is: creating it, through a cell read from a slot, or through `SharedGC`'s built-in
   pointers.
