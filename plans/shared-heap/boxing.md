# Boxing Pointers to the Shared Heap

Date: 2026-08-24
Revised: 2026-11-24
Status: Partly implemented; see the table below. Built only with `HERMES_ENABLE_SHARED_HEAP`.
Relates to: [marking.md](marking.md) (the marking protocol: P-rules and channels),
[barrier.md](barrier.md) (the store barrier and the rings), [allocator.md](allocator.md) (the
allocation lock and black allocation)

---

## Summary

A normal heap refers to a shared cell only through a **box**: a `BoxedSharedObject`, one per runtime
per shared cell. It is a `JSObject` subclass holding a raw `SharedGCCell *` in a plain C++ field,
which the local GC does not trace.

Each runtime keeps a map from shared cell to box, `Runtime::sharedObjectBoxes_`, with two jobs:

* **Deduplication.** At most one live box per cell per runtime, so `===` works on shared objects.
* **Roots.** Every shared cell the runtime references is a key in its map (dead keys stay until
  pruned), so all runtimes' maps together are the shared heap's root set. A box needs no refcount,
  root list or finalizer.

Design decisions:

1. **The box is a real `JSObject` subclass:** the arm64 JIT reads `clazz` and `parent` of any
   `Object`-tagged value after only a tag test, and `vmcast` is unchecked in release builds, so
   anything else would be a silent wild read.
2. **The box is opaque:** it is never unboxed automatically; planned hooks forward property access.
3. **One box per cell per runtime is required:** two compiled forms of `===` cannot be hooked.
4. **The box has a null parent,** as the spec requires, and cannot gain one. It keeps a box that is
   the receiver out of the negative-match cache (Appendix A.4).

**Riskiest gaps:** `{...box}` and `JSON.stringify(box)` silently return `{}` on high-traffic paths
(Appendix C). Once property hooks exist, a box can poison the read inline cache unless the
descriptor-flag skip lands with them (Appendix A.4, O-B8).

| Mechanism | Status |
|---|---|
| The box: cell kind, hidden class, vtable, `BuildMeta` | implemented |
| Box follow-ups: descriptor flag, `isBoxedSharedObject()` | not yet implemented |
| The map: weak-root walk, prune, map lock | implemented |
| `SharedGC::makeBoxed`: allocate a shared cell and its box | implemented; only tests call it |
| The collector's root scan, `SharedGC::markRuntimeRoots` | implemented; `collect()` calls it, and only tests create a `SharedGC` |
| Registering a runtime at construction, deregistering it at teardown (handing its keys to the collector if marking is on) | implemented |
| The materialiser (P4), `Runtime::materializeSharedSlot` | implemented; only tests call it |
| The prune push (P6) | implemented; only tests trigger the prune |
| JS-visible hooks (Appendices A to C) | not yet implemented |
| Shared strings (Appendix D) | not yet implemented; partly open |
| The in-flight check before the root scan | implemented, two passes (marking.md P1) |

No production code creates a shared cell yet: no JS or JSI API makes one, `makeBoxed`, the
materialiser and `SharedGC::collect` run only in tests, and only tests create a `SharedGC`, with or
without its collector thread. `HERMES_ENABLE_SHARED_HEAP` is off by default; enable it with
`-DHERMES_ENABLE_SHARED_HEAP=ON` in CMake or `-c hermes.enable_shared_heap=true` in Buck. Off, it
compiles out `lib/VM/SharedHeap/`, `BoxedSharedObject` and the `Runtime` glue.

### Terms

* **Marking is on** while `cycleSeq_` is odd (`SharedGC::markingOn()`).
* **Black allocation:** every new shared cell is born marked (P8).
* **Ring:** a runtime's local list of pushed cells (barrier.md). A full ring **spills**: its entries
  move to the collector's barrier worklist under `barrierMtx_`.
* **Barrier site:** code that tests whether marking is on and acts on the answer. Its **window**
  runs from just before the `cycleSeq()` test (in the materialiser's window, the slot read comes
  after the test) through the effect the test guards: the slot store for the store barrier (P5),
  publishing the box for the materialiser (P4), the push for the prune (P6), and the operation
  itself for an `Atomics` read-modify-write. A runtime is **inside a barrier** while it is inside a
  window. The materialiser's hit check is not a site: it reads the slot outside any window, only as
  a lookup key.
* **In-flight check** (implemented; marking.md P1, barrier.md "The in-flight check"): each runtime
  keeps a counter that is odd while it is inside a barrier. After turning marking on, the collector
  reads each runtime's counter and, if it is odd, waits until it changes. It then makes a seq_cst
  read-modify-write of `cycleSeq_` that leaves the value unchanged, and reads every counter and
  waits again. Every window either is waited for, or found already closed, by one of these two
  passes, so its effect happens before the root scan, or tests the flag after the read-modify-write
  and so synchronizes with it.
* **Materialiser:** reads a shared slot and returns this runtime's box for the cell in it, creating
  and publishing the box if needed.
* **Overlap slots:** the first direct-property slots of a `JSObject`, which a subclass reuses for
  its own C++ fields. `BuildMeta` tells the local GC how many to skip.
* **Exotic descriptor:** a property descriptor for a host-object, proxy or (planned) box property.
  Its `slot` holds the property's raw `SymbolID`, not a slot index.
* **v1:** the first shipped version of shared structs.
* **Locks:** `mtx_` is the shared heap's allocation lock. `barrierMtx_` guards the runtime registry,
  the barrier worklist and ring drains. The map lock, `sharedObjectBoxesMtx_`, guards one runtime's
  map structure.

---

## 1. The box

### 1.1 The `CellKind`

Implemented: `CELL_KIND(BoxedSharedObject)` follows `FastArray` in `CellKinds.def`. It must stay
inside `CELL_RANGE(Object, JSObject, NativeJSFunction)`, since `JSObject::classof` is a range test,
and before `CELL_KIND(BoundFunction)`, or `Callable::classof` misclassifies it. It cannot go last
(`CellKind.h` asserts `CallableExpectsThisKind_last == AllCellsKind_last`), and it must not be a
`CELL_CLASS`, which adds a predefined string and shifts every later `SymbolID`.

### 1.2 The hidden class

Implemented: `boxedSharedObjectClass`, a strong root, is a root class of its own with a null
parent (a `JSObject` keeps its parent in its hidden class), built in
`Runtime::initRootHiddenClasses` with the box's overlap slots reserved, so a box never shares a
class with an ordinary object. The class is not extensible, so a box can gain neither a prototype
nor a property (tests `Create_ReservesTheOverlapSlots` and `Create_CannotGainAPrototype`).
**Detect a box with `vmisa<BoxedSharedObject>`, not by hidden-class identity.**

### 1.3 `ObjectVTable`

Implemented with the `JSObject` defaults for the seven indexed entries, as in `JSDate`. Writing
`ObjectVTable{VTable(k, s)}` would compile and leave them null. There is no finalizer: the map's
prune makes one unnecessary.

### 1.4 `BuildMeta`

```cpp
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
```

Implemented. **The pointer is untraced only because of the overlap-slot count.** Computing it as
`numOverlapSlots<JSObject>()` (0) would pass the builder's assert and make the local GC decode the
shared address as a `GCSmallHermesValue`. In HV64 (the CMake default) a raw address reads as a
double, which the GC skips, so nothing visible happens. With HV32 on a 64-bit host, the address's
high word can carry a pointer tag, and the GC then corrupts it or crashes. No test catches this
reliably: `Bound_SurvivesALocalGCWithItsCellIntact` binds a made-up address that decodes as a
non-pointer in every mode. The `static_assert` is the guard: a count too small to cover
`sharedCell_` does not compile. Because the pointer is never a `HermesValue`, HV32's compressed
encoding, which would truncate it, never sees it.

### 1.5 Instance size

`cellSizeImpl` floors at `cellSizeJSObject()`, i.e. `sizeof(JSObjectAndDirectProps)`:

| Config | `cellSizeJSObject()` | overlap slots | usable direct slots | **allocation** | Debug |
|---|---|---|---|---|---|
| HV32 (compressed pointers + boxed doubles), 64-bit host | 40 | 2 | 4 | **40 B** | 48 B |
| HV64 | 80 | 1 | 5 | **80 B** | 88 B |
| HV32 on a genuine 32-bit device | 40 | 1 | 5 | **40 B** | 48 B |

**A box costs a full empty `JSObject`, 40 or 80 bytes per (runtime, shared cell) pair**, and cannot
be smaller; the pointer fits in the overlap slots. Do not hard-code the usable slot count. The
first two rows were measured with `sizeof`, out of six direct slots; the last is arithmetic.

### 1.6 Telling a box apart: a descriptor flag and a kind test

* **Per descriptor, a flag fits.** `PropertyFlags` uses 12 of 16 bits, so `sharedObject : 1` is
  free. Every hook in Appendices A and B keys off it. Not yet implemented.
* **Per object, no flag fits.** `SHObjectFlags` is nine flags plus `objectID : 23`. Taking a bit
  would halve the object-hash space, and the field is also the lazy-object index and part of the
  Static Hermes C ABI. Instead add `isBoxedSharedObject()`, a `vmisa` test, beside `isHostObject()`.
  It costs about the same, since `kind_` is in the cell's first word. Not yet implemented.

Keep class tests off the descriptor path, where the existing tests read one bit of a flags word that
is already loaded. A class test there adds a load and a compare to every step of every prototype
walk.

### 1.7 Remaining work on the box

`sharedCell_` is bound once (`setSharedCell` asserts it was null) and never retargeted, since a
retarget would be a new acquisition needing its own barrier. Not yet implemented, and none of it
fails the build if skipped:

* the descriptor flag and `isBoxedSharedObject()`;
* the read-cache guard (O-B8): the existing read-path asserts pass for a box, and the JIT has none;
* the hooks of Appendices A to C;
* the `PrLoad`/`PrStore` invariant (Appendix A.5).

---

## 2. The map: deduplication and roots

### 2.1 Container

`llvh::DenseMap<SharedGCCell *, WeakRoot<BoxedSharedObject>> sharedObjectBoxes_`, a `Runtime`
member. Implemented. The `WeakRoot` keeps the map from holding a box alive. Accessors:
`findBoxForSharedCell` (lookup), `setBoxForSharedCell` (insert), and `appendSharedRootKeys` and
`appendSharedRootKeysUnlocked` (copies of the keys, with and without taking the map lock).

### 2.2 The local GC's weak-root walk

Implemented. `Runtime::markWeakRoots` walks the map on the mutator thread with this runtime's world
stopped. It calls `acceptWeak` on every value and tests death with `getNoBarrierUnsafe`;
`WeakRoot::get`'s read barrier would revive the box. It never erases. It sets
`sharedObjectBoxesNeedPrune_` when a value became null.

**The walk must stay outside `if (markLongLived)` and visit every entry on every weak-root pass**,
young- or old-generation, dead or not. The GC updates these `WeakRoot`s only here, so a skipped
entry keeps a stale pointer to a box that moved or died, and a later lookup returns it (test
`Entries_AreVisitedOnEveryWeakRootPass`).

**The walk takes no map lock.** Stopping this runtime's world does not stop the shared collector,
but the walk writes only values, never keys, bucket occupancy or the table size, and keys and values
are separate memory locations. So no local GC pause ever waits on the collector, and a background
collector cannot cause a priority inversion with a UI-thread young-generation pause.

### 2.3 Pruning: flag dead entries during GC, erase at the next lookup or box allocation

Implemented. When `sharedObjectBoxesNeedPrune_` is set, `pruneSharedObjectBoxesIfNeeded` calls
`pruneSharedObjectBoxes`, which erases every dead entry under the map lock. `findBoxForSharedCell`
calls it before it looks up, and `makeBoxed` and the materialiser after they allocate their box.

* **Not in the weak-root walk.** The walk also runs from debug heap checks
  (`HadesGC::checkWellFormed`, on by default in Debug builds) and from heap snapshots, with
  acceptors that must not change anything, and erasing does not belong in a stop-the-world pause.
* **Never in the insert** (test `Insert_DoesNotPrune`). `makeBoxed` inserts while holding `mtx_`,
  and the prune push (P6) can spill under `barrierMtx_`, against the lock order, so `makeBoxed`
  prunes before it takes `mtx_`. Dedup needs only the lookup's prune: the materialiser looks up
  before it inserts, and `makeBoxed`'s cell is new. The duplicate-box assert in
  `setBoxForSharedCell` uses a plain `find` for the same reason, and because an assert must have no
  side effects: `findBoxForSharedCell` prunes and fires the read barrier.
* **Allocating a box prunes too** (O-B14). With lookups alone, a runtime that allocates with
  `makeBoxed` and never materialises would never prune, so every key it inserted would stay a root
  after its box died, and its map and the heap would only grow. So `makeBoxed` prunes after
  allocating the box and before taking `mtx_`, where no lock is held (test
  `MakeBoxed_PrunesWhatALocalGCFoundDead`). The materialiser prunes between its box allocation and
  its window, so a prune that allocation flags does not walk the whole map with the window open
  (test `Materialize_PrunesBeforeTheWindow_AndPushesBothKeys`). A retry after exhaustion needs a
  prune too (allocator.md, "Collecting before throwing").
* **Nothing prunes outside tests:** only tests call `makeBoxed` and the materialiser, which make the
  only lookups.

**The prune push (P6): implemented** in `Runtime::pruneSharedObjectBoxes`.

* **What it does:** when marking is on, the prune pushes each dying key onto the runtime's ring
  before erasing it.
* **Why:** soundness needs it. A dead box does not make the key safe to drop: while the box was
  alive, the runtime may have stored the cell into a cell born this cycle, which is never traced. If
  the key is then pruned before the root scan copies the map, only this push marks the cell
  (marking.md P6 has the sequence). It also softens a breach of the raw-pointer rule without making
  it safe: a raw pointer that outlived its box stays valid only until the sweep of the first cycle
  that begins after the prune, unless something else keeps its cell, and a later use can dangle.
* **Order:** push every dying key, then take the map lock and erase. A root scan that misses an
  erased key took the map lock after the prune, so it comes after the push, and the collector's
  later ring drain sees it.
* Skip the push when `sharedMutator_` is null.
* The prune is a barrier site, and every prune runs outside any window and opens its own: in the
  hit check, in `makeBoxed` before the lock, and in the materialiser before its window. The lookup
  inside that window has nothing left to prune, since nothing between there and the prune before it
  can run a local GC.

### 2.4 `DenseMap` hazards

* `SharedGCCell` must be a complete type where the map is instantiated, since `DenseMapInfo<T *>`
  uses `alignof(T)`. `Runtime.h` includes `SharedGCCell.h` for this.
* Erasing during iteration is legal (it leaves a tombstone and does not rehash); inserting is not.
* **Allocate the box, then insert.** The allocation can run a local GC, which rewrites values and
  may set the prune flag. Hold no iterator or bucket reference across it, and insert no placeholder
  with a null `WeakRoot`: it looks dead, and the next lookup erases it.
* **Insert with `operator[]`.** `insert` keeps an existing dead entry and leaves the new box
  unregistered: dedup breaks, and once the entry is pruned the live box's cell is not a root.

### 2.5 Memory accounting

Design decision: the map is not counted in `Runtime::mallocSize()`, which counts only SH units and
the identifier table. The map is the group's bookkeeping, not this runtime's. On a 64-bit host it
costs 21-43 B per entry depending on load factor (11-21 B on 32-bit), and the first insert allocates
64 buckets (1 KiB; 512 B on 32-bit).

### 2.6 How a runtime gets a shared pointer: allocation and acquisition

**Every shared cell a runtime references is boxed and in its map.** That makes the maps the complete
root set: apart from black allocation, tracing and ring pushes, the collector finds cells only
through keys copied from the maps. A runtime gets a `SharedGCCell *` only by **allocation**
(`SharedGC::makeBoxed`, implemented) or **acquisition** (the materialiser, implemented), and both
box. Migrating a String or BigInt into the shared heap (planned, through the proposed `storeField`,
Appendix B.3) is an allocation, so it will be boxed too. The raw allocation routes, which skip the
map, exist only in unit-test builds (O-B12). A raw slot read (`getPointerOrNull`, `getPointer`,
`SharedArray::at`) skips it in every build, and may be used only as the raw-pointer rule below
allows.

**The raw-pointer rule** (marking.md, "Roots"). A raw `SharedGCCell *` is valid only while a box for
its cell is held by a strong root, such as a `PinnedValue` in registered `Locals`. The map is weak,
so being in it is not enough. Shared cells never move, so the only hazard is being freed. The rest
of this section covers the two cases in which a raw pointer exists before its box is published.

#### Allocation: `makeBoxed`

**The gap.** Between allocating a cell and inserting it into the map, the cell is in no map, and
black allocation does not cover the start of a cycle: `beginCycle` clears every mark bit and turns
marking on, then the root scan copies the maps. A cell allocated just before that and published
after its map was copied is unmarked and unscanned, so it is swept while its runtime holds it. The
collection runs on another thread, so reordering the steps on this thread cannot close the gap.

**The fix: box first, then one critical section.**

```
box = BoxedSharedObject::create(runtime)    // may run a local GC; not in the map yet
{ hold mtx_
    cell = allocate the shared cell         // born marked
    box->setSharedCell(cell)                // no allocation
    runtime.setBoxForSharedCell(cell, box)  // takes the map lock; may malloc, never GCs
}
```

* **Holding `mtx_` excludes `beginCycle`'s mark reset and turn-on**, which run together under
  `mtx_`. So the cell is either published before marking turns on, and the later root scan finds it,
  or allocated after, born marked, and stays marked.
* **Allocating the box first makes holding `mtx_` safe.** Nothing in the section allocates on a GC
  heap, so no local GC runs while every runtime's shared allocation is blocked on `mtx_`.

Details:

* An oversized request fails before the box is allocated. If the cell allocation fails (budget
  exhausted or mapping failure), the box already allocated becomes garbage. Either way `makeBoxed`
  returns an empty `PseudoHandle`.
* The box is rooted in `makeBoxed`'s own `Locals` and returned as a `PseudoHandle` for the caller to
  root, since the map holds it only weakly. Rooting it in the caller's `GCScope` would pin one dead
  box per out-of-memory retry.
* `SharedGC.h` only declares `makeBoxed`. The definition is in `SharedGCBoxing-inline.h`, so nothing
  that includes `SharedGC.h` has to include `Runtime.h`. The rule is about headers: `SharedGC.cpp`
  includes `Runtime.h` for the root scan, which is allowed.
* **A box must be able to exist before its cell**, so code that walks boxes must accept a null
  `sharedCell_` (tests `Bind_CanHappenAfterConstruction` and `Unbound_SurvivesALocalGC`). If a box's
  class or layout were ever derived from its cell's shape, box-first would not work for that shape:
  the box would have to be created generic and given its class only when bound (a two-phase box).
* The insert can `malloc` under `mtx_` when the map grows: no GC, but unbounded work. `reserve()`
  alone does not bound it, because prune tombstones can force a rehash at the same size.
* **The runtime must be attached to this heap**, registered and not yet detached. `makeBoxed`
  publishes into whatever map it is given, and the root scan reads only registered runtimes' maps:
  for a runtime created without this `SharedGC`, attached to another, or past its post-shutdown
  deleter, no cycle of this heap reads the key, so its next cycle frees the cell while the box
  still names it, unless a traced cell reaches it. Another heap's collector would also mark and
  trace a cell it does not own. So `makeBoxed` asserts `Runtime::isAttachedTo`, which checks that
  the runtime's `SharedGC` is this one and that it still has its `SharedMutator`.

#### Acquisition: the materialiser (implemented)

Implemented as `Runtime::materializeSharedSlot` (marking.md P4 and P11): check for a live map hit
outside any window; on a miss, allocate a box and prune, then, inside one window, test the marking
flag, re-read the slot with an acquire load (O-B10), push the cell if marking is on, look it up
again, and bind and publish with `operator[]`. It returns a live map hit through the barriered
`WeakRoot::get`, never `getNoBarrierUnsafe`, and returns a `PseudoHandle`, empty if the slot holds
no cell. In unit-test builds an overload also takes a hook, which the body, in `Runtime.cpp`, calls
after the box allocation and before the prune, between the entry and the test, between the test and
the read, after the read, and just before the window closes; `SharedMaterializeTest` uses it to
replay CE-1 (marking.md) and the orderings below. The materialiser's runtime must be attached, and
its caller must not be inside a window or hold `mtx_` or the map lock, and must hold a box for the
cell that contains the slot across the call: the slot is read in place, and a cycle can run while
the box is allocated (the raw-pointer rule). A field-read hook has its receiver's box; code that
reached the containing cell through another slot must box that cell first.

**The gap to close.** A miss path that held a raw pointer across `BoxedSharedObject::create`, which
can run a local GC, would leave this gap: runtime A reads cell C while marking is off and starts
allocating a box. Runtime B overwrites the slot, removing the last edge to C, and records nothing
because marking is off. A collection starts and scans roots before A publishes: C is in no map and
unreachable, so it is swept, and A then publishes a dangling key. One runtime alone cannot cause
this: A finishes publishing before it can run a store that removes the edge.

**The shape: a hit check outside any window, then one window after the box allocation.** One window
around the whole materialiser would also close the gap, since the collector would wait until the box
is in the map, but it would span `BoxedSharedObject::create`, which can run a local GC, and the
collector would wait through that GC. A window that ends before `create` reopens the gap.

1. **Hit check, outside any window:** read the slot only as a key for the lookup, and look the cell
   up. On a live hit, return that box. Its key protects the cell, so it needs no push (marking.md
   P4), and since nothing between the read and the lookup allocates and a live key's cell cannot
   have been freed, the box is the one for the value read. The value is never followed: with no
   window, another runtime can overwrite the slot with nothing recorded, and a cycle can sweep the
   cell after the read.
2. **On a miss:** allocate the box, outside any window, then prune if that allocation's local GC
   found a box dead, so the lookup inside the window has nothing to prune.
3. **The window:** test the marking flag, then re-read the slot. If the slot no longer holds a
   pointer, drop the box and return an empty `PseudoHandle`. Otherwise push the cell if marking is
   on, before taking the map lock, then look up the cell it re-read. On a live hit, return that box
   and drop the new one; on a miss, bind and publish the new box with `operator[]` before the window
   closes. Nothing in this window allocates on a GC heap.

On a miss, only the read inside the window is used, because a whole cycle can pass between the hit
check and the window. This mirrors `makeBoxed`, with the window in place of `mtx_`. The window needs
four orderings:

* **A seq_cst entry,** as at every site: with a release entry, the in-flight check and a window that
  tests too early can both miss each other.
* **A seq_cst test before the re-read.** A test that reads the read-modify-write that starts the
  in-flight check's second pass, or a later value, synchronizes with it, so the re-read cannot
  return a value overwritten by a store whose barrier tested the flag before marking turned on. A
  re-read made before the test could, unless it were itself seq_cst; if that was the cell's last
  edge, the cell would be swept. After the test, an acquire re-read is enough, and publication needs
  one anyway (O-B10).
* **The push before the map lock.** A push can spill under `barrierMtx_`, which the root scan takes
  before a map lock.
* **The publish inside the window.** A pass that waits for the window, or finds it already closed,
  then also sees the key.

With these, the collector finds the key of every materialiser whose window one of the passes waits
for or finds already closed: the first pass for one whose test saw marking off, and the second pass
for one whose test saw marking on before the read-modify-write. A materialiser whose test came after
the read-modify-write may open its window after the second pass looks, and its push can then land
after certification and be dropped by its stamp. That push is not needed: either the slot held the
cell at the cycle's snapshot (marking.md), which the cycle keeps, or a runtime stored the cell there
later, and that runtime still protects it.

So no push the materialiser makes is needed, as far as bounded GenMC models of this shape can show,
under RC11 and IMM: without the push no execution loses a cell, while a release entry, a weaker or
later test with a re-read that is not seq_cst, a publish after the window closes, a single pass, or
a missing read-modify-write between the passes loses one with the push or without it. The push
before the map lock is a lock-order rule, which the models, having no spills, do not test. The push
is kept as insurance while the argument rests on bounded models.

**Rejected: a per-runtime in-flight root,** holding the raw pointer where the collector can see it
while the box is allocated. It saves the second read, but the root scan and a leaving runtime's key
copy would both have to read the new root, and it still needs a short window for the read itself.

### 2.7 The collector's read: one lock per runtime

Implemented. Each runtime's map has its own lock, `sharedObjectBoxesMtx_`:

| Operation | Thread | Takes the map lock? |
|---|---|---|
| Weak-root walk (`markWeakRoots`) | owner, in a local GC pause | **no**: writes values only |
| Lookup (`findBoxForSharedCell`) with no prune | owner | no: a read on the only writing thread |
| Insert (`setBoxForSharedCell`) | owner | yes: `operator[]` can rehash |
| Prune (`pruneSharedObjectBoxes`) | owner | yes for the erase, which leaves tombstones; not for the pushes before it, which only read |
| Root scan (`SharedGC::copyRootKeys`) | collector | yes: tried under `barrierMtx_`, then held alone across the copy |
| Key copy by `unregisterMutator` (`appendSharedRootKeysUnlocked`) | the thread detaching the runtime | no: nothing inserts or erases while a runtime detaches, and the root scan only reads |
| Wait by `unregisterMutator`, after the unlink | the thread detaching the runtime | yes, released at once: it waits for a root scan's copy |

`SharedGC::markRuntimeRoots` copies every registered runtime's keys (`copyRootKeys`) and marks the
copies after releasing every lock. It snapshots the registry under `barrierMtx_`; then, for each
runtime, it takes `barrierMtx_` again, checks that the runtime is still registered and tries its map
lock, and copies the keys under the map lock alone. A busy map is retried after a yield: waiting for
it while holding `barrierMtx_` would stall every spill for as long as an insert or a prune holds the
map, and waiting without `barrierMtx_` would let the runtime be destroyed first. The map lock keeps
the map alive through the copy: `unregisterMutator` takes it, and releases it at once, after the
unlink, so it waits for a copy begun while the runtime was registered, and no later scan can find
the runtime. Dead keys not yet pruned are copied too, which over-retains and never under-retains. A
dead key can linger: a young-generation pass does not null an old-generation box, so a dead promoted
box is nulled only when the next old-generation collection completes its marking
(`completeMarking`), and its key stays a root until the next prune. A runtime becomes visible to the
scan when it registers, at the end of its constructor (`Runtime::create` takes an optional
`SharedGC`), once its map exists. Internal JS runs before that, which is safe because it cannot
reach shared cells.

**Copying one map at a time** is meant to be sound, provided the materialiser keeps its decided
shape. A key added after its map was copied came from one of two places:

* From `makeBoxed`: the cell was born after marking turned on, so it is marked.
* From the materialiser: if its test saw marking on, it pushed the cell (P4); a push that lands
  after certification is dropped by its stamp and is not needed ("Acquisition: the materialiser").
  If its test saw marking off, it was inside a window when marking turned on, and the in-flight
  check makes the collector wait for that window, so its key is in the map before the copy.

A runtime that registers after the scan's snapshot need not be scanned, which is the same case: its
map is empty when it registers, so every key in it was added after the snapshot.

The materialiser case relies on that shape: its hit check follows no pointer, it reads the pointer
it publishes only inside its window and keeps that window's orderings, and the collector makes both
passes of the in-flight check. Nothing enforces the shape: `setBoxForSharedCell` does not check that
the materialiser calls it inside its window. A key erased before its map is copied is pushed first
(P6), and a runtime that leaves before its map is copied hands its keys to the barrier worklist as
it unregisters (marking.md P12).

**Lock order** (the full order is in marking.md, "State and locks"): `barrierMtx_` then the map lock
(the root scan, which only tries it); `mtx_` then the map lock (`makeBoxed`'s insert); `barrierMtx_`
before `mtx_` if both are ever held (declared in `SharedGC.h`; nothing nests them today). Never the
map lock then `barrierMtx_`, so the prune and the materialiser push first. Never `mtx_` then
`barrierMtx_`, so an insert never prunes, and `makeBoxed` prunes before it takes `mtx_`.

**Cost: a short stall, not a pause.** While a runtime's keys are copied, its insert or prune waits.
`makeBoxed`'s insert waits holding `mtx_`, so other runtimes' shared allocations wait too. The copy
is O(map size) with no tracing, and a local GC never waits. No map is walked under `barrierMtx_`:
the root scan holds it only to find a runtime and try its map lock, and a leaving runtime copies its
own keys before it takes it (O-B3). A spill can still wait for that runtime's append of the copy to
the barrier worklist, a memcpy made only while marking is on. A detaching runtime may wait for a
root scan's copy of its map, holding no lock.

---

## 3. Identity

### 3.1 Dedup is required, not an optimisation

`===` on two objects is decided by raw comparison in seven places, two of them unhookable:

| Site | Hookable? |
|---|---|
| `strictEqualityTest`: raw compare, then tag compare, then string or BigInt contents | yes |
| `_sh_ljs_strict_equal` | yes |
| JSI `HermesRuntimeImpl::strictEquals(const jsi::Object &, const jsi::Object &)` | yes |
| C ABI `strict_equals_object` (`hermes_vtable.cpp`): compares the two `JSObject` pointers | yes |
| `_sh_ljs_strict_equal_inline`: returns early on `a.raw == b.raw` | partly |
| **arm64 JIT inline**: a raw compare when either operand is an object or a non-pointer; the slow call comes only after the string-tag test | **no** |
| **shermes AOT**: `canCompareStrictEqualityRaw` is true for two statically-Object operands, so `===` becomes a C `==` with no call | **no** |

**Every path that produces a box must go through the map**, including slow-path hooks and JSI value
conversion.

### 3.2 Raw-pointer identity outside `===`

Searching for `strictEqualityTest`, `isSameValue` and `isSameValueZero` finds the value-equality
sites: the equality opcodes, `Object.is`, the `defineOwnProperty` and Proxy invariants, Map and Set
keys, and `includes` and `indexOf`. JSI `strictEquals` and the C ABI's `strict_equals_object` do not
call them; they compare raw pointers (3.1). The search also misses three sites that compare raw
pointers and would need an unwrap if dedup were ever weakened: `ordinaryHasInstance` (which
`instanceOfOperator_RJS` calls), `JSObject::setParent`'s prototype-cycle check, and
`Runtime::insertVisitedObject`, the cycle guard of `Array.prototype.join` (which `toString` calls)
and `toLocaleString`, where **duplicate boxes cause unbounded recursion**.

### 3.3 Do not derive `objectID` from the shared address

Doing so would be legal, since `objectID` is a 23-bit hash whose users re-check with `isSameValue`,
and it would keep the hash stable when a box is recreated. But it fixes nothing: `OrderedHashMap`,
`JSWeakMapImpl` and the `structuredClone` memo match entries by raw pointer anyway. So not in v1.
Regardless, a box must never set `flags_.lazyObject`, which makes `objectID` the lazy-object index,
and a derived `objectID` must not be 0, since `getAlreadyAssignedObjectID()` asserts that it is set.

### 3.4 Identity is observable through GC

A box can die and be recreated, so a shared object's identity is stable only while its box is
reachable:

```js
wm = new WeakMap();
wm.set(o.p, 1);   // box B1 for shared cell S, held only weakly
// ... a local GC frees B1, the prune drops the map entry, and the WeakMap entry goes
wm.get(o.p);      // a fresh box B2 for the same S -> undefined
```

`WeakRef.deref()` changes the same way, and a `FinalizationRegistry` can fire for an object still in
use. A strong `Map` key keeps the box alive, and `Object.is(o.p, o.p)` within one expression is
safe, because the first result is held in a register root. **v1 rule:** identity holds only while
the box is reachable, and `WeakMap`, `WeakSet`, `WeakRef` and `FinalizationRegistry` reject shared
objects with a `TypeError`. TC39 has not decided whether shared structs may be `WeakMap` keys, and
pinning a box that becomes a weak key stays possible later. Not yet implemented: `canBeHeldWeakly`
is the natural place, but JSI `createWeakObject` and the C ABI's `create_weak_object` skip it.

---

## 4. The JS-visible surface, in brief

A box is a real `JSObject` with an empty hidden class, so generic operations do not crash; they
**succeed with the wrong answer**. None of the hooks exists yet. Reads go through two descriptor
functions and the slot accessors (Appendix A). Writes go through `putNamedWithReceiver_RJS` and its
computed mirror, which `Object.defineProperty` and private fields bypass (Appendix B). Everything
else, from enumeration to `structuredClone`, needs the same follow-through or a throw (Appendix C).

To find sites, search for `isHostObject()`, `flags_.hostObject`, `desc.flags.hostObject` and
`isProxyObject()`. That misses the `ObjectVTable` entries (O-B7) and `Runtime::insertVisitedObject`
(3.2), which have no host or proxy branch to copy.

---

## 5. Rejected alternatives

* **Local slots holding a direct (compressed) pointer to a shared cell.** In HV32,
  `_sh_cp_encode_non_null` truncates a shared address with no range check, and decoding yields a
  live local address. The design also needed a shared pointer base, `memfd`/`MAP_FIXED` work and
  address space taken from the local heap. With boxing, no local slot names a shared cell.
* **`addCustomWeakRootsFunction` for the map walk.** It is for weak roots that are not `Runtime`
  members. It adds a `std::function`, and with no `removeCustomWeakRootsFunction` the closure would
  have to outlive the runtime's last GC.
* **An allocation entry point that takes a publish callback.** "Nothing under the lock may allocate"
  would become an unchecked contract on the caller's callable. `makeBoxed` has one body, and that
  body is the contract.
* **A map lock held from allocation through publishing.** It would be held across the box
  allocation, and so possibly a local GC, which the root scan would then wait out, and with it the
  cycle. If it were also held across the lookup, the prune would re-lock the non-recursive map lock
  and take `barrierMtx_` after it, against the lock order.
* **One global map lock.** Every runtime's map updates would wait on every other runtime's.
* **A separate root log.** Not needed: the map entries are the root set.

---

## 6. Open questions

Most important first: O-B8 and O-B11. The rest follow in number order.

**O-B8. The read-cache guard**, deferred to the interpreter work. Open: the
`!desc.flags.sharedObject` skip (Appendix A.4) has not been prototyped. Once the A.1 hook lands, it
is what keeps both read-cache entries clean; the null parent (implemented) covers only the
negative-match entry when a box is the receiver. One guard is enough: `negMatchClazz` is written
only in `JSObject::getNamedWithReceiver_RJS`. `GET_BY_ID_IMPL`, the AOT `getByIdWithReceiver_RJS`
and the JIT's `emitParentSpecialization` only read it, and the JIT uses its parent specialization
only when it is set.

**O-B11. The materialiser's gap** (implemented in the decided shape): a hit check outside any
window, then one window after the box allocation; a published in-flight root is rejected (both in
"Acquisition: the materialiser"). It relies on the in-flight check's two passes (marking.md P1),
argued on paper and checked by bounded GenMC models of the design, not the code. The argument that
copying one map at a time is sound depends on the materialiser keeping this shape. Open: nothing
enforces the shape; `setBoxForSharedCell` does not check that the materialiser calls it inside
its window (marking.md, "Open questions").

**O-B1. 32-bit atomicity.** `SharedValue` holds a `std::atomic<uint64_t>`, so it cannot tear. Open:
whether that atomic is lock-free on every 32-bit target Hermes ships to. The comment on
`kBitsPerWord` in `SharedGC.h` says it is not; that is unverified (ARMv7-A has `LDREXD`/`STREXD`,
and x86-32 has `CMPXCHG8B`), and the comment should be corrected once this is settled. If a target
is not lock-free, slot accesses stay correct but each takes a lock; narrowing the slot there to 4
bytes with its own tagging would avoid that, if the narrower slot keeps the same memory orders
(O-B10).

**O-B2. Uniqued strings are excluded in v1** (settled), structurally under the Appendix D end state.
`obj[sharedStr] = 1` still works by copying into the identifier table, which already happens for
every non-uniqued string: the `canBeUniqued()` reuse path in `getOrCreateIdentifier` is commented
out pending an `isLongLived` GC API. The real cost is that a shared identifier table is ruled out.

**O-B3. How long the root scan holds `barrierMtx_`** (settled). Only long enough to find each
runtime and try its map lock, never across a copy, so no `spill()` waits for one ("The collector's
read: one lock per runtime"). The map lock, taken while the runtime is registered and held across
the copy, keeps a detaching runtime's map alive: deregistration takes it after the unlink. A leaving
runtime copies its own keys before it takes `barrierMtx_`, and appends them under it while marking
is on.

**O-B4. Teardown.** Implemented: a runtime deregisters (resets its `SharedMutator`) in a
post-shutdown deleter. `~Runtime` runs the shutdown callbacks, which may allocate, run JS and
trigger GC, before `finalizeAll()` and the deleters, so deregistering in a callback could be
followed by new boxes. Once a runtime is deregistered, its map is not read and its keys stop being
roots; one that deregisters while marking is on first copies its keys into the barrier worklist, so
that cycle still marks them (marking.md P12). Also implemented: `~SharedGC` stops the collector
thread, which finishes any cycle it is running. No runtime is attached by then, so nothing is a
root, and the destructor runs one more collection, whose sweep frees every cell left, before it
asserts the heap is empty and releases the regions. A cycle that was already in progress would not
do: its marks reflect roots that existed when it began. So no cycle may be in progress on the
destroying thread. Each runtime holds a `shared_ptr` to its heap, so when a runtime holds the last
reference, all of this runs inside its `~Runtime`, on the thread destroying it, after it has
detached: that thread joins the collector thread, if the heap has one, waiting for any cycle it
is running, and then runs a whole collection (`Teardown_InTheLastRuntimeFreesWhatItBoxed`, whose
heap has no collector thread).

**O-B5. Enumeration semantics**, consistent across `JSObject::getOwnPropertyKeys`,
`JSObject::forEachOwnPropertyWhile` (used by spread) and the `HiddenClass::forEachProperty*` calls
in the `JSON.stringify` walker and the snapshot edge emitter.

**O-B6. `writable` on a box's fields.** Undecided: what `Object.getOwnPropertyDescriptor(box, 'x')`
returns for `writable`, and whether `Object.freeze(box)` throws. The answer may turn on a defect in
the TC39 draft's handling of `writable` that is not written down here; re-derive it from the draft
before deciding.

**O-B7. `SharedArray` indexed hooks.** The seven `ObjectVTable` entries are its whole indexed
implementation and are not designed. `FastArray` is the template, but its compile-time
`lengthPropIndex()` does not port. The box needs `indexedStorage`, which only constructors set, or
`box[i] = v` becomes a named property; but the materialiser allocates the box before it may read the
cell's kind ("Acquisition: the materialiser"). Open: set the flag when the box is bound, inside the
window, or pick the box before the allocation, as D.4 does for views. The store underneath exists:
`SharedValue::set` and `initField`.

**O-B9. Where `isBoxedSharedObject()` goes:** about 83 existing `isHostObject()`/`isProxyObject()`
decision points, plus the Appendix C sites that have no host or proxy branch.

**O-B10. Publication ordering.** Implemented. `SharedValue::set`'s slot store is seq_cst, so it is a
release, and every slot read that may be followed to a cell is an acquire: the copy constructor (so
`SharedArray::at`), `getPointer`, and `getPointerOrNull`, which the tracer uses. So a runtime that
reads a new cell's address from a slot sees the cell as it was built, birth mark included, and so
does the collector, which reads the mark bit of a cell it reaches through a slot, and its kind and
length if it traces it. The store is seq_cst rather than only a release, and the store barrier's
pre-read is a seq_cst load, because the barrier also needs both in one total order with the turn-on
and every test of the marking flag (marking.md P7). `isEmpty` and `isPointer` are relaxed, since
nothing is read through their answer, and so is `initField`, which is only for a cell nothing else
can reach.

**O-B12. Raw allocation routes** (settled). `SharedGC::makeA`, `SharedArray::create` and
`BoxedSharedObject::create(runtime, cell)` bypass the map, so they are declared only in `UNIT_TEST`
builds; tests rely on them. Production code allocates through `makeBoxed` and acquires through the
materialiser, both of which create an unbound box with `BoxedSharedObject::create(runtime)` and bind
it themselves. The public box constructor stays, because `makeAFixed` needs it.

**O-B13. Shape registry and symbolization.** Field lookup (Appendix B.2) needs a shared shape
registry and a per-runtime symbolization table. Neither is designed.

**O-B14. Pruning without a lookup** (settled). `makeBoxed` and the materialiser prune after
allocating their box, so a runtime that allocates and never materialises still drops its dead keys
("Pruning: flag dead entries during GC, erase at the next lookup or box allocation"). Still open: a
retry after exhaustion needs a prune too (allocator.md, "Collecting before throwing").

**O-B15. A cell in transit between runtimes.** Channel 5 (marking.md) makes a handoff legal only
while the sender keeps its box alive until the receiver has boxed the cell. `postMessage` cannot
promise that: the Worker extension queues a serialized message for as long as the receiver takes,
while the sender's JS may drop the box and the next prune its key. A message that carries a
shared cell (Appendix C, row 39) needs a root of its own until the receiver has boxed the cell:
either a strong reference to the sender's box, released on the sender's thread once delivery is
known, or a group-wide table of cells in transit, which the root scan copies and whose removal
pushes while marking is on, as the prune does (P6). Neither is designed, and the second reopens "A
separate root log" (section 5) for messages. The receiver also needs a way to box a cell it was
handed: `materializeSharedSlot` boxes only what it reads from a slot. And Worker runtimes cannot
join a group yet: the extension creates them with `makeHermesRuntime`, which takes no `SharedGC`.

---

# Appendix A — The read path

Not yet implemented.

## A.1 Named access

Hook **`JSObject::getNamedDescriptorUnsafe`**, in the own-object branch and in the matching branch
of the prototype walk, modelled on its `hostObject` branch (below). Open: where the walk tests for a
box. With no per-object flag (1.6), the test is `isBoxedSharedObject()`, and running it on every
object visited adds a load and a compare to every step of every walk. A box's parent is null, so the
test could run only at the object that ends the walk, provided `Object.setPrototypeOf(box, x)`
throws (Appendix C, row 14). The A.3 hook has the same choice. The `hostObject` branch:

```cpp
if (LLVM_UNLIKELY(selfHandle->flags_.hostObject)) {
  desc.flags.hostObject = true;
  desc.flags.writable = true;
  desc.slot = name.unsafeGetRaw();
  return *selfHandle;
}
```

Then dispatch in **`JSObject::getNamedWithReceiver_RJS`**, at its
`!accessor && !hostObject && !proxyObject` guard and beside its
`vmcast<HostObject>(propObj)->get(name)` call. `getNamed_RJS` forwards to it, and every engine's
slow path ends there:

| Engine | Path |
|---|---|
| Interpreter | the `GetById` slow path → `getNamed_RJS` |
| Interpreter, with receiver | → `getNamedWithReceiver_RJS` |
| AOT / SH | `_sh_ljs_get_by_id_rjs` → `getById_RJS` → `getByIdWithReceiver_RJS` → `getNamedWithReceiver_RJS` |
| arm64 JIT | `DECL_GET_BY_ID(getById, "getById", _sh_ljs_get_by_id_rjs)` → same as AOT |

JSI and JSLib reach it too. Not hooked: `doGetByIdSlowPath_RJS` (its fast path is duplicated in
`getByIdWithReceiver_RJS`, so hooking it would mean writing the branch twice) and `GET_BY_ID_IMPL`.
Both gate on conditions a box never meets, so a box falls through to the hook.

## A.2 The slot-accessor family

On an exotic descriptor, `desc.slot` holds the raw `SymbolID`, not a slot index. The slot accessors
in `JSObject.h` that take a descriptor handle this in one of two ways:

* Five check at run time, behind a `proxyObject || hostObject` guard, and need only
  `|| desc.flags.sharedObject`: both `getNamedSlotValue` overloads, `setNamedSlotValue`,
  `getComputedSlotValue` and `setComputedSlotValue`. They then re-enter the generic get or put,
  which reaches the hook.
* The `Unsafe` forms only assert: `getNamedSlotValueUnsafe` and `setNamedSlotValueUnsafe` directly,
  and `getComputedSlotValueUnsafe` and `setComputedSlotValueUnsafe` through them. Release builds
  check nothing. Extend the asserts with `!desc.flags.sharedObject`, and make sure every caller
  excludes boxes, as callers already exclude host and proxy objects.

Direct slot readers that test only some flags:

* The debugger's `displayName` lookup tests only `propObj`, and a shared struct may declare
  `displayName`. It uses `getNamedSlotValue`, so the accessor fix covers it.
* `HermesBuiltin`'s `@@iterator` lookup tests only `proxyObject` before `getNamedSlotValueUnsafe`.
  It is safe only because shared field names are strings. Add the flag.
* `JSError`'s stack-property path tests `!accessor && !proxyObject && !hostObject`. Add the flag.
* The `@@iterator`-on-`JSArray` paths (interpreter slow paths and `Operations.cpp`) use
  `getNamedSlotValue`, so the accessor fix covers them.

So a box claims only the names its shape declares, not every name as HostObject does: each claimed
name is one more place where a `SymbolID` can be used as a slot index.

## A.3 Computed and indexed access

Second hook: **`JSObject::getComputedPrimitiveDescriptor`**, beside its `hostObject` and
`proxyObject` branches and before the prototype step. `box[k]` and computed destructuring go through
it, as do `getComputed_RJS` and `putComputed_RJS`. Its consumers need the flag too:
`getComputedWithReceiver_RJS`, `hasComputed` and `putComputedWithReceiver_RJS`. The fast paths in
front of it require `fastIndexProperties`, which a struct box lacks, and the JIT has no computed
inline cache. Hermes has no `Atomics` object yet; how `Atomics.*` reaches a field is open
(marking.md, "Open questions").

Indexed access uses the `ObjectVTable` entries instead, and only for an object with
`indexedStorage`, which a struct box lacks. For `SharedArray`, see O-B7.

## A.4 The read inline cache and its negative-match entry

A read cache entry is keyed by hidden class, and all boxes share one class, so an entry written for
one box would be used for every box. An entry has two parts:

* **The positive entry** (`cacheEntry->clazz` and `setSlot`) is written only when the property was
  found. Today the boxes' shared class has no properties, so it is never a positive key. Once the
  A.1 hook lands, a box returns found descriptors whose slot is a raw `SymbolID`.
* **The negative-match entry** (`negMatchClazz`) is for a property found on the receiver's immediate
  prototype. `JSObject::getNamedWithReceiver_RJS` sets it to the **receiver's** class:

```cpp
if (selfHandle->getParent(runtime) == propObj &&
    !selfHandle->getClass(runtime)->isDictionary()) {
  ...
  cacheEntry->negMatchClazz = selfHandle->getClassGCPtr();
}
```

The interpreter's `GET_BY_ID_IMPL`, the AOT `getByIdWithReceiver_RJS` and the JIT's
`emitParentSpecialization` then read the parent's slot directly. Without a null parent, another box
at the same site would get the prototype's value instead of its own field. When a box is the
prototype, the cached slot is a raw `SymbolID` used as a slot index: any name whose ID is at most
0xffff passes the `desc.slot <= kMaxSlot` check, so the read is wild. The interpreter and AOT copies
assert `!proxyObject`, `!hostObject` and `!lazyObject`; **a box passes those asserts and is still
wrong**. The JIT has no assert, bakes the result into compiled code and never calls back into C++.

* **A private class is not enough:** all boxes would share it, so an entry written for box A would
  match box B.
* **A null parent** (implemented) protects the negative-match entry when a box is the receiver:
  `negMatchClazz` is written only when the receiver's parent is `propObj`, and readers check
  `parentGCPtr` first. It does nothing for the positive entry, or for a box used as a prototype
  (`{__proto__: box}`, legal and exercised by V8's tests).
* **The descriptor-flag skip** covers the rest. The whole cache-population block is inside the
  `!accessor && !hostObject && !proxyObject` branch, so adding `&& !desc.flags.sharedObject` keeps
  both entries clean. This is why follow-through must be a descriptor flag. Not yet implemented
  (O-B8).

`Runtime::getNamed` and `Runtime::putNamedThrowOnError` keep class-keyed caches with no exotic
check. They are safe only because filling them requires a real own property.

## A.5 What bypasses the hook

| Path | Bypasses? |
|---|---|
| `_sh_ljs_get_by_id_rjs_inline` | No: falls through to the out-of-line call |
| JIT generic inline cache | No: branches to the slow path |
| JIT `emitObjectSpecialization` | No: compares `hiddenClassLazyJITId` |
| JIT `emitParentSpecialization` | **Yes**: A.4, handled by the null parent and the descriptor-flag skip |
| `PrLoad`: HBC `GetOwnBySlotIdx`, shermes `_sh_prload` | **Yes**: a raw direct-slot read with no class check, cache or call |
| `PrStore`: HBC `PutOwnBySlotIdx` and its shermes equivalent | **Yes**: the same, writing |

`PrLoad` and `PrStore` are emitted when the type checker knows an object's class layout. A typed
checked cast tests only that a value is an object, so a box passes a cast to a class type, and a
`PrStore` can then overwrite `sharedCell_`, which lives in the overlap slots. **v1 invariant: a box
must never reach code typed with a user class.** Nothing enforces it yet; an explicit assert is
worth adding.

---

# Appendix B — The write path

Not yet implemented.

## B.1 Hook sites

**Named: `JSObject::putNamedWithReceiver_RJS`**, two edits:

1. The direct-slot guard (`!accessor && !internalSetter && !hostObject && !proxyObject && writable`)
   gains `!desc.flags.sharedObject`. Otherwise the raw `SymbolID` in `desc.slot` is used as a slot
   index and overwrites property storage.
2. A dispatch branch beside `vmcast<HostObject>(...)->set(...)`. Its gate
   (`selfHandle != receiver || isHostObject() || isProxyObject()`) must admit boxes too.

Without both, the put reaches `addOwnProperty`: the box gains a real own property, leaves the shared
class for good, and diverges from the shared cell. All three engines get here through
`putNamed_RJS`, and the JIT's `PutById` has no inline path. Computed puts need the same two edits on
the mirror.

Other routes reach `addOwnPropertyImpl` without `putNamedWithReceiver_RJS`:

* `Object.defineProperty`, through `defineOwnPropertyInternal` (whose exotic test covers only lazy
  and proxy objects), `defineNewOwnProperty` and `defineOwnComputedPrimitive`.
* **Private fields:** `AddOwnPrivateBySym` calls `defineNewOwnProperty`, so a base-class constructor
  that returns a box lets a subclass install private fields on it.

The spec has one choke point: the shared struct's `[[Set]]` calls `[[DefineOwnProperty]]`. Hermes
has five entry points (named put, computed put, `defineOwnPropertyInternal`,
`defineOwnComputedPrimitive`, `defineNewOwnProperty`), so the check must be one helper called from
all five.

The fast path before the call is inert: `tryGetOwnNamedDescriptorFast` returns `llvh::None` when the
boxes' class has no property map and `false` when it has one, since the map holds only the reserved
overlap slots; both callers require `hasValue() && getValue()`.

## B.2 `CanBeSharedAcrossAgents` and the store

A proposal; not prototyped. In the TC39 draft, a shared struct's `[[DefineOwnProperty]]` throws
`TypeError` when `CanBeSharedAcrossAgents(v)` is false (the operation itself never throws), and
`s.x = v` reaches it through `[[Set]]`. The box's write branch:

1. **Resolves the field** in the shape's name-to-index table, through the per-runtime symbolization
   table (O-B13). An unknown key returns false in sloppy mode and throws `TypeError` in strict mode;
   it never falls through to `addOwnProperty`.
2. **Checks the value:**

   | Value | Action |
   |---|---|
   | `undefined`, `null`, Boolean, Number | store inline in the 64-bit shared slot; `SharedValue` has no encoding for these yet (marking-implementation.md open question 3; [shared-gccell.md](../../doc/plans/shared-heap/shared-gccell.md) open question 1) |
   | String | **migrate into the shared heap** (an allocation, so boxed), then store the address |
   | BigInt | migrate; `BigIntPrimitive` is already eligible |
   | Symbol | **throw `TypeError`**: a deliberate v1 divergence; the TC39 draft admits Symbol |
   | `vmisa<BoxedSharedObject>` | **unbox**: store `box->sharedCell_`, never the box pointer |
   | any other Object | throw `TypeError` |

3. **Stores** through the single slot-writing function (B.3).

**GC discipline.** Migrating a String or BigInt allocates, so this path can trigger a GC. Hold no
raw `NamedPropertyDescriptor`, `JSObject *`, slot pointer or unrooted `SharedGCCell *` across it.
`FastArray` encodes a value before checking capacity for the same reason.

## B.3 Unboxing on the way in

**A shared slot must never hold a pointer to a box.** That would be a shared-to-local edge, which
breaks the one-way graph that marking depends on, and it would dangle after a local GC that the
shared collector cannot see.

**Exactly one function turns a JS value into a shared slot value and stores it** (proposed:
`SharedObject::storeField(runtime, cell, index, HermesValue)`). It checks, unboxes and calls
`SharedValue::set`. A private slot type with that function as its only friend enforces this cheaply.
Today `SharedValue::set` and `initField` are public.

**The read side is the materialiser** (marking.md channel 1): every shared-object reference loaded
from a slot goes through the map before it becomes a `HermesValue`. These two functions carry most
of this design's correctness.

**Open: immediates.** B.2 stores `undefined`, `null`, Booleans and Numbers inline, but `SharedValue`
has no encoding for them yet (open question 1 of `doc/plans/shared-heap/shared-gccell.md`):
`getPointerOrNull`, which the tracer and the materialiser use, decodes every non-zero slot as a
cell. A field read also needs one slot read, inside the window, that yields either the immediate or
the box. `materializeSharedSlot` returns an empty handle whenever a read finds no cell and drops
what it read, so a field read built on it reads the slot twice, and a store between the two can
change it from an immediate to a pointer or back.

## B.4 Write-side caches

* `JSObject::tryCacheAddProperty` skips proxies and host objects because "what they hold is not
  described by their HiddenClass, so the epoch cannot track them". Skip boxes too. Its loop starts
  at the parent and never inspects `self`, which is safe only because the B.1 hook keeps boxes out
  of `addOwnProperty`.
* Three add caches (inline caches that remember the hidden-class change made by adding a property)
  would fire for every box at a site, since boxes share a class and a parent: the interpreter's
  `PutById` cache fill, the AOT equivalent, and the write cache in `_jit_put_by_id`.

---

# Appendix C — The generic object surface

Not yet implemented.

What happens **today** when a box reaches each operation (it behaves as an empty `JSObject` with a
null parent), what it **must** do, and the risk: **C** silently wrong at high traffic, **H**
silently wrong, **M** wrong but loud or rare, **L** cosmetic or diagnostic. Row numbers are stable.

## C.1 Read and query

| # | Operation | Where | Today | Must | Risk |
|---|---|---|---|---|---|
| 1 | `box.x` | `getNamedWithReceiver_RJS`, null descriptor | `undefined` | field value | **C** |
| 2 | `box[k]` | `getComputedWithReceiver_RJS` | `undefined` | field value; a **separate hook**, A.3 | **C** |
| 3 | `'x' in box` | `JSObject::hasNamed` (tests only `proxyObject`) | `false` | `true` for declared fields | **H** |
| 4 | `'x' in box`, computed | `JSObject::hasComputed` | `false` | `true` | **H** |
| 5 | `Object.getOwnPropertyDescriptor` | `objectGetOwnPropertyDescriptor`; its HostObject branch is the shape needed | `undefined` | a descriptor; see O-B6 | **M** |
| 6 | `hasOwnProperty` / `Object.hasOwn` | `objectPrototypeHasOwnProperty`, which has a host branch | `false` | `true` | **M** |
| 7 | `propertyIsEnumerable` | `objectPrototypePropertyIsEnumerable` | `false` | `true` | **L** |
| 8 | `Reflect.get`/`has`/`ownKeys` | delegate to `JSObject::` | **free** | — | — |

## C.2 Mutation

| # | Operation | Where | Today | Must | Risk |
|---|---|---|---|---|---|
| 9 | `box.x = v` | `putNamedWithReceiver_RJS`, null descriptor → `addOwnProperty` | **adds a real own property; the box leaves the shared class** | write the shared field | **C** |
| 10 | `box[k] = v` | computed mirror → `addOwnProperty` | same | same | **C** |
| 11 | `Object.defineProperty` | `defineOwnPropertyInternal` tests lazy/proxy → `addOwnProperty` | succeeds | throw, or define the field | **H** |
| 12 | `delete box.x` | `JSObject::deleteNamed`: `if (!pos) { if (!lazyObject && !proxyObject) return true; }` | returns **`true`**, changes nothing | **`false`** | **H**, the most misleading |
| 13 | `delete box[k]` | `JSObject::deleteComputed` | `true` | `false` | **H** |
| 14 | `Object.setPrototypeOf(box, x)` | `JSObject::setParent`; only `Object.prototype` and the global object have an immutable prototype | **succeeds** | `TypeError` | **M** |
| 15 | `freeze`/`seal`/`preventExtensions` | `JSObject::freeze`/`seal`/`preventExtensions` | succeeds; the state is per box and lost when the box is recreated | see O-B6 | **M** |
| 16 | `isSealed`/`isFrozen` | `JSObject::isSealed`/`isFrozen` | `true` once `noExtend` is set | per the TC39 draft | **L** |
| 17 | `getPrototypeOf`, `__proto__` | `JSObject::getParent` | `null` | `null` | — |

## C.3 Enumeration: three places, not one

| # | Operation | Where | Today | Risk |
|---|---|---|---|---|
| 18 | `Object.keys`/`values`/`entries` | `JSObject::getOwnPropertyKeys`: its exotic branches cover only lazy and proxy objects; HostObject has a **separate** branch calling `getHostPropertyNames` | `[]` | **C** |
| 19 | `Object.getOwnPropertyNames` | same function as row 18 | `[]` | **C** |
| 20 | `Object.getOwnPropertySymbols` | `objectGetOwnPropertySymbols` | `[]`, which is correct: no symbol fields | **L** |
| 21 | `Object.assign` | `objectAssign`: keys from `JSObject::getOwnPropertyKeys` (row 18), then `JSObject::getOwnComputedDescriptor`, which has no host or box branch; the `\|\| from->isHostObject()` test is in `HermesBuiltin`'s spread slow path, which `Object.assign` does not use | copies nothing | **H** |
| 22 | `{...box}` | **`HermesBuiltin`'s copy-data-properties helper**: its only guard is `source->isProxyObject() \|\| source->isHostObject() \|\| getMayHaveAccessor()`, so a box takes the `forEachOwnPropertyWhile` fast path | copies nothing, silently | **C** |
| 23 | `JSON.stringify(box)` | the object walker tests only `isProxyObject()`, then takes a `HiddenClass`-only fast path through `forEachProperty` | `{}` | **C** |
| 24 | `for (k in box)` | `getForInPropertyNames` and its helper `appendAllPropertyKeys`, in `JSObject.cpp`; the cache is read and written (`clazz->setForInCache`) only when `shouldCacheForIn` (`!isDictionary() && !indexedStorage && !hostObject && !proxyObject`) | `[]`, **cached on the class all boxes share** | **C** |
| 25 | `Object.defineProperties(t, box)` | `objectDefineProperties` | does nothing | **M** |
| 26 | JSI `getPropertyNames` | → `getForInPropertyNames` | `[]` | **M** |
| 27 | for-in re-validation | the interpreter and AOT `GetNextPName` paths | — | **M** |
| 28 | `setProtoClasses` / `matchesProtoClasses` | already special-case `head->isProxyObject()` | caches | **M** |

**Row 24 gets worse once enumeration works.** All boxes share one class, and `shouldCacheForIn`
excludes host and proxy objects but not boxes, so a for-in cache written for one box is read for
every box. Today it holds `[]`. Once rows 18 and 24 are fixed, one box's field list would be served
for boxes of every shape. Exclude boxes in `shouldCacheForIn`.

## C.4 Coercion, identity and typing

| # | Operation | Where | Today (null parent) | Risk |
|---|---|---|---|---|
| 29 | `String(box)`, `` `${box}` `` | `toPrimitive` → `ordinaryToPrimitive`, which tries `toString`, then `valueOf`, then raises "Cannot determine default value of object" | **`TypeError`**: correct and loud | — |
| 30 | `typeof box` | `typeOf` | `"object"` | — |
| 31 | `box instanceof X` | `ordinaryHasInstance` | `false` | — |
| 32 | `Object.prototype.toString.call(box)` | `objectPrototypeToString`; reading `@@toStringTag` on the box is a free hook | `"[object Object]"` | **L** |
| 33 | `box.toString()`, `box.hasOwnProperty(…)` | no prototype | `TypeError: not a function`, the strongest argument for a null parent | — |
| 34 | `{__proto__: box}` / `Object.create(box)` | legal and exercised by V8's tests | lookup must read through to the shared cell; puts a box in a prototype chain → A.4, B.4 | **C** |
| 35 | `Array.prototype.join`/`toString` cycle guard | `Runtime::insertVisitedObject` | duplicate boxes → **unbounded recursion** | **H** |
| 36 | prototype-cycle check | `JSObject::setParent` | raw pointer compare | **M** |

## C.5 Native surfaces

| # | Operation | Where | Today | Risk |
|---|---|---|---|---|
| 37 | JSI `setNativeState` / `hasNativeState` | reject Proxy and HostObject; no box branch | **succeeds**, and the state is lost when the box is recreated | **M** |
| 38 | JSI `setExternalMemoryPressure` | rejects Proxy only; no box branch | succeeds | **L** |
| 39 | `structuredClone` / `postMessage` | `SerializedValue` raises "Host Objects are not serializable"; no box branch | serializes a box as an **empty object** | **H** |
| 40 | `WeakMap`/`WeakSet`/`WeakRef`/`FinalizationRegistry`; JSI `createWeakObject`; C ABI `create_weak_object` | `JSWeakMapImpl` keys on `gcStableHashHermesValue` plus a `WeakRef`; the JSI and C ABI calls skip `canBeHeldWeakly` | **nondeterministic**: see "Identity is observable through GC" | **H** |
| 41 | Heap snapshot edges | `BoxedSharedObject::vt` has the default snapshot metadata, with no edge callback | the shared cell is invisible | **L** |
| 42 | Snapshot node name | the same default metadata names the node by its cell kind | the cell kind's name | **L** |
| 43 | CDP / DevTools | the remote-object converters call JS `Object.getOwnPropertyNames` and `getOwnPropertyDescriptor` | follows from rows 5 and 19 | **L** |
| 44 | `tryGetNamedNoAlloc` | stops its walk at a proxy or host object | walks past a box and answers "not found" | **L** |

The fix for rows 41 and 42 belongs in `BoxedSharedObject::vt`'s snapshot metadata, not in
`JSObject.cpp`. The TODO in `BoxedSharedObject.cpp` suggests a native-memory edge.

**Ranked by likelihood times silence:** rows 22 (`{...box}`) and 23 (`JSON.stringify`) first, with
the most traffic and no signal; then 18 and 24 (enumeration); then 2 (`box[k]`, easy to forget once
`box.x` works); then 9 (`box.x = v`, which takes the box off the shared class and fills the add
cache wrongly for every other box at that site); then 12 (`delete` returning `true`).

---

# Appendix D — Shared strings

Not yet implemented. D.1 is the v1 plan; D.2 to D.5 describe a possible end state; D.6 is open.

The proposal has exactly **one** String type: `CanBeSharedAcrossAgents` admits every string ("If
val is a String, return true", under "Primitives can always be shared"). "Shared string" only says
where the characters live; nothing observable may distinguish one.

## D.1 v1: strings take the Object slow path

**Decision: a shared string is a `BoxedSharedObject` like any other, unboxed at use.** This closes
marking.md's channel 3: a string box is in the same map, rooted the same way and covered by the same
acquisition push. It turns a potential GC soundness hole into a **conformance gap**, which tests can
find; `typeof` is the canary. It is a scope cut, not a design:

* **Coverage is all-or-nothing.** With one String type, any missed site is an observable deviation.
* **Three sites cannot be reached from C++.** The arm64 JIT compiles the two hottest string
  predicates as inline tag tests: `Emitter::typeOfIs`/`jmpTypeOfIs` emit `emit_sh_ljs_is_string`,
  and `Emitter::strictEqualImpl`/`jStrictEqual` branch to not-equal on a tag mismatch before the
  `_sh_ljs_strict_equal` slow call. So `box === "abc"` compiles to `false`, and dedup cannot help,
  as it does for box against box. Third, JSI presents a box as an object to **out-of-tree** code:
  `valueFromHermesValue` checks `hv.isString()` before `hv.isObject()`, and `jsi::Value::isString()`
  reads a `kind_` fixed at conversion. TurboModules branch on it, and Hermes cannot patch them.

Not blockers: plain `typeof x` reaches the out-of-line `typeOf` in every tier, the interpreter
directly and AOT and JIT code through `_sh_ljs_typeof`; shermes is hookable on both predicates,
since `canCompareStrictEqualityRaw` is false when both operands can be strings; and
`String.prototype`'s 39 methods need one coercion hook, because they use `checkObjectCoercible` and
`toString_RJS`, not `isString()`.

## D.2 End state: a Str-tagged view cell

A per-runtime `StringPrimitive` subclass holding an untraced raw `SharedGCCell *` to characters in
the shared heap. Being Str-tagged, it passes every existing "is this a string?" test with no edits;
being an ordinary local cell, it needs nothing from the local collector.

**Kind placement: a UTF16/ASCII pair right after `BufferedASCIIStringPrimitive`.** Constraints,
hardest first:

1. **Outside `SymbolStringPrimitive::classof`'s range** (`DynamicUniquedUTF16` to `ExternalASCII`).
   `canBeUniqued()` is `vmisa<SymbolStringPrimitive>`, and `convertToUniqued` writes `weakUniqueID_`
   behind only an `assert(canBeUniqued())`, so a view in that range would write past its end in
   release builds. This is what makes O-B2 structural.
2. Inside `CELL_RANGE(StringPrimitive, …)`, which an interior insertion satisfies with no macro
   edit.
3. Below `ExternalUTF16StringPrimitiveKind`, so `isExternal()` stays a one-sided `>=`. After
   `ExternalASCIIStringPrimitive`, `isExternal()` would become a range test, adding a comparison to
   the first test of both cast functions, on every character access.
4. Kinds keep alternating UTF16, ASCII, which `isASCII()`'s parity test relies on. A pair inserted
   inside an existing pair goes ASCII first.

Four interior positions meet all four: after each of `DynamicUTF16`, `DynamicASCII`, `BufferedUTF16`
and `BufferedASCII`. Use the last, so the new branch goes **last** in the cast chain. Only
`StringPrimitive::classof` widens, for free, through `CELL_RANGE`.

**Shape.** 24 B in HV64 and HV32-on-64-bit, 16 B on genuine 32-bit: `GCCell`, `lengthAndFlags_`,
`hash_` and the pointer. It must use `makeAVariable` with a VTable `size` of 0, as
`BufferedStringPrimitive::create` does, because `cellSize<>` static-asserts against
`VariableSizeRuntimeCell`. `BuildMeta` is a bare `setVTable`; registering `sharedCell_` would make
the local GC decode a shared address. No finalizer, `creditExternalMemory` or `trimSize_`.

**Three required edits:**

* **Extend the two 8-element `cellKindsContiguousAscending` lists** in `isASCII()` and
  `isExternal()` to ten. Forgetting fails a `static_assert`, the safe kind of failure.
* **Add a branch to `castToASCIIPointer` and `castToUTF16Pointer`.** Their final `else` is an
  unchecked `vmcast<BufferedASCIIStringPrimitive>`, whose `getRawPointer()` reads `concatBufferHV_`
  at the offset `sharedCell_` would occupy. Without the branch, a raw shared address is decoded as a
  `GCHermesValue`. Debug builds catch it (`vmcast` is `llvh::cast`); release builds do not.
* **Add `assert(!vmisa<SharedView…>(this))` to both `castTo*PointerForWrite` functions.** Their
  final `else` is an unchecked `vmcast<Dynamic*>` with a fixed-offset `getRawPointerForWrite()`, so
  a view there overwrites its own `sharedCell_` and then adjacent objects. An assert, not a real
  test, because `StringBuilder::appendCharacter` calls these once per character.

## D.3 Cost of the view on string access

The cast functions are an if/else chain over one loaded header word: `isExternal()`, then
`vmisa<DynamicUniqued…>`, then `vmisa<Dynamic…>`, then an unchecked `vmcast<Buffered…>`. With the
view branch **last**, External still takes one test, DynamicUniqued two and Dynamic three: **the
fast-path cost is zero**. Only Buffered strings, at or above `CONCAT_STRING_MIN_SIZE`, pay one more
compare. Bulk operations hoist the dispatch through `StringView`, whose iterator holds the raw
pointer, so only about seven sites dispatch per character. One is hot (`getByValTransientFast`, the
interpreter's `s[i]` fast path), and for local strings it still takes the Dynamic branch at today's
instruction count.

## D.4 Rooting, and why dedup still matters

**Widen the map value and change nothing else:** `WeakRoot<BoxedSharedObject>` becomes
`WeakRoot<GCCell>`. `acceptWeak` takes `WeakRootBase &`, so this is free. One map, not two: the
shared cell's kind selects exactly one local representation, and the hit check, which never follows
the pointer, finds either one with a single lookup. On a miss the materialiser allocates the
representation before its window, and may follow the pointer only inside it ("Acquisition: the
materialiser"), so it must pick the representation before it can read the cell's kind: from a tag in
the slot's encoding, which `SharedValue` does not have yet, or from the kind read in a window of its
own before the allocation. When the cell it re-reads in the publishing window has the other kind, it
drops the representation and starts again. Box and view cannot share a base class, because
`CELL_RANGE(Object, …)` and `CELL_RANGE(StringPrimitive, …)` are disjoint.

**Views still need dedup.** String equality compares contents, so a duplicate view is invisible to
`===`, but it is not harmless. The map is also the root set, with one `WeakRoot` per key. If a
second, unregistered view exists when the registered one dies, the local GC nulls the root, the
prune erases the key, the cell is freed, and the second view reads freed characters, with no
finalizer or barrier to catch it. For strings, dedup is a memory-safety requirement rather than an
identity one.

## D.5 What the shared payload should be

**Not a `StringPrimitive`:** an immutable `{length, encoding, hash, chars[]}` blob with no mutable
header. Then `lengthAndFlags_` exists only in local views, and lazy-state hazards (hash caching,
in-place representation changes) become impossible rather than forbidden. The hash write lands in
the view's local header, but the view's constructor must copy the precomputed hash and set its flag,
or every runtime recomputes it. Uniquing, the GC clearing the uniqued bit and `weakUniqueID_`'s read
barrier are unreachable because of the placement rule.

**Every string operation copies out of the view.** `concat`, `slice`, `getOrCreateIdentifier` and
`StringView::getUTF16Ref` go through the cast functions and copy, so `sharedStr + t` makes a full
local copy. Views pay off for strings that are stored, read and compared, not manipulated.

## D.6 Open

1. **Memory arithmetic, which could reverse the recommendation.** Per (runtime, shared string), a
   view costs 24 B **plus** a map bucket of 21-43 B (64-bit). A local `DynamicASCIIStringPrimitive`
   copy costs about 16 B of header plus the characters, about 32 B for 16 characters, and Hermes
   strings are mostly short. Ignoring the shared payload, which the group pays once, break-even is
   about 29-51 ASCII characters (15-26 UTF-16). A likely answer is **view above a length threshold,
   copy below**, which also caps map growth. Compute this before committing; it is the strongest
   case for copying.
2. **BigInt.** `1n` is `Tag::BigInt`, neither Object- nor Str-tagged, so the view design does not
   cover it. A view BigInt is harder: `BigIntPrimitive` is `final`, its `classof` tests one kind,
   and `getDigits()` calls `getTrailingObjects` directly, so there is **no dispatch layer to
   extend**. Either leave BigInt out of v1 and price copying `1n` back, or cost a third view kind.
