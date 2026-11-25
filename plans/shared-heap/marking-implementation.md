# Implementing the Shared Marker

Date: 2026-08-27
Revised: 2026-10-21
Status: The tracer, the root scan, the wait before tracing (two passes of the
in-flight check), the prune push (P6), the materialiser with its acquisition
push (P4), `collect()`, the collector thread that calls it and teardown are
implemented and unit-tested. Only unit tests create shared cells or run a
collection.
Relates to: [marking.md](marking.md) (rules P0-P12, numbered channels),
[barrier.md](barrier.md) (barrier, rings, the wait, certification),
[boxing.md](boxing.md) (box maps), [allocator.md](allocator.md) (mark bits,
black allocation),
[shared-gccell.md](../../doc/plans/shared-heap/shared-gccell.md)
(a shape-driven tracer that would replace the kind switch)

---

## Summary

The tracing core of the shared-heap collector: how it finds a cell's slots,
how it marks and queues cells, where roots and ring entries enter, and the
order of one collection. The rules are in marking.md; the barrier, the wait
and certification are in barrier.md.

| Piece | Status |
|---|---|
| Slot visitor, `SharedGC::markCell`, mark acceptor, mark worklist | Implemented |
| Root scan (`markRuntimeRoots`) and `SharedGC::collect` | Implemented |
| The wait before tracing (the in-flight check) | Implemented, two passes; argued on paper and checked by bounded GenMC models of the design, not the code (open question 1) |
| Publication ordering: a seq_cst slot store in `SharedValue::set`, acquire slot loads | Implemented |
| The prune push (P6), and a leaving runtime's keys copied to the barrier worklist (P12) | Implemented |
| The materialiser and its acquisition push (P4) | Implemented; only tests call it |
| A collector thread and its trigger; teardown | Implemented; only tests create a `SharedGC` |
| A debug check after each certified cycle (marking.md, "A safety net for P10") | Implemented, in `HERMES_SLOW_DEBUG` builds |
| Immediates; a termination safety net for release builds | Open |

Terms:

* **Marking is on** while `cycleSeq_` is odd (`markingOn()`). Turning marking
  on or off adds one to it.
* **Box**: a `BoxedSharedObject`, through which one runtime's JS refers to a
  shared cell. A runtime's box map, `sharedObjectBoxes_`, maps cells to boxes.
* **Materialiser**: `Runtime::materializeSharedSlot`, which reads a shared
  slot and boxes the cell it finds. Only tests call it.
* **Push**: append a cell to the runtime's own ring, stamped with the
  `cycleSeq_` value the barrier's test returned.
* **Window**: the span of a barrier site from just before its test of
  `cycleSeq_` through the effect the test guards. A runtime's in-flight
  counter is odd while it is inside one (marking.md P1).
* **Ring**: a runtime's 256-entry buffer of pushed cells (`SharedBarrierRing`).
  A full ring spills into `MarkState::barrierWorklist` under `barrierMtx_`.
* **Black allocation**: every new cell is born marked, even between cycles.
* **SATB** (snapshot-at-the-beginning): while marking is on, the store barrier
  records each value a store overwrites, so a cell reachable at the start of
  the cycle is not lost when its last edge is removed.
* **Certify**: a round certifies when one observation of the barrier worklist
  and every ring, made under `barrierMtx_` after the trace, marks nothing new;
  marking then turns off in the same critical section (marking.md P10,
  termination; barrier.md, "Certification").

---

## How one collection runs

`SharedGC::collect()` runs these steps on one thread, holding `cycleMtx_`
throughout, while runtimes keep running. The collector thread calls it
(allocator.md, "Budget, mappings and exhaustion"), `~SharedGC` calls it once
more at teardown, and so do tests. The caller must hold no shared-heap or map
lock and must not be inside a barrier window, or the cycle would wait for the
caller forever.

1. **Start** (`beginCycle`). Holding `barrierMtx_`, create `markState_`.
   Then, holding the allocation lock `mtx_`, clear every mark bit and turn
   marking on. The lock keeps black allocation's `fetch_or` from racing the
   clear (a plain `memset`), and keeps a cell from being born between the two:
   it would keep its mark and not be traced this cycle, so a value stored into
   it before marking turned on would have no record (marking.md P8, black
   allocation and the reset). Last, holding `barrierMtx_` again, note every
   runtime whose in-flight counter is odd.
2. **Wait** until no barrier that read `cycleSeq_` before marking turned on
   can still perform its effect: call `pollInFlightBarriers()` until it
   returns true, yielding in between. See "Waiting before tracing".
3. **Wait again** (`beginSecondInFlightPass`): make a value-preserving
   `cycleSeq_.fetch_add(0, seq_cst)`; then, holding `barrierMtx_`, note every
   runtime whose in-flight counter is odd, and poll `pollInFlightBarriers()`
   until it returns true again. The write hands what the first wait learned
   on to every barrier that tests `cycleSeq_` after it. See "Waiting before
   tracing".
4. **Scan roots** (`markRuntimeRoots`; see "Roots").
5. **Trace and certify**: repeat `drainMarkWorklist()` and `certifyRound()`
   until a round certifies (barrier.md, "Certification").
6. **End** (`endCycle`): destroy `markState_`. In `HERMES_SLOW_DEBUG` builds,
   then check that every cell reachable from a box-map key is marked and still
   allocated.
7. **Sweep** (`sweep`), on the same thread. Mark bits are set with relaxed
   atomics; sweeping on the thread that traced orders the sweep after every mark
   that thread set, by program order (allocator.md, "What publishes the marks to
   the sweeper"). In `HERMES_SLOW_DEBUG` builds, `collect` then checks that no
   box-map key names a freed cell.

Never sweep after `abortCycle`. It turns marking off and drops both worklists,
so unmarked no longer means unreachable. The next `beginCycle` clears the
partial marks; until then a debug flag makes `sweep()` assert.

---

## Finding a cell's slots

As in Hades, finding a cell's slots is separate from what the collector does
with each one. `SharedSlotVisitor::visit(acceptor, cell)` switches on
`cell->getKind()` and calls the `detail::visitSlots` overload for that kind.
The acceptor is a template parameter supplying `accept(SharedGCCell *)`;
nothing is virtual.

* **The kind is the only thing to dispatch on.** A shared cell has no
  `VTable` and no `Metadata` entry. Shared kinds are listed after every local
  kind, so their values fall past the end of the kind-indexed tables rather
  than aliasing an entry.
* **The switch is generated** by including `CellKinds.def` with only
  `SHARED_CELL_KIND` defined, so a new shared kind without a `visitSlots`
  overload fails to compile instead of going untraced. The call is qualified,
  so every overload must be declared above `visit()`.
* **`SharedArray` walks `length()` slots**, never `SharedGC::allocatedSize()`:
  the size bucket rounds up, and the slack past `end()` was never initialised.
* **One acquire load per slot.** `getPointerOrNull()` decodes a slot in one
  load and returns null for an empty slot, which is skipped. Another runtime
  may be storing to the slot, so `isPointer()` then `getPointer()` would be
  two loads of a value that can change in between. The load is an acquire,
  and pairs with the seq_cst store in `SharedValue::set`, which is also a
  release. It does two jobs. It makes the cell's construction visible:
  `accept` sees a new cell's birth mark, and a later visit sees the kind and
  length the cell was built with (boxing.md O-B10). And it orders
  termination without a fence: `set`'s store releases the push its barrier
  made, the next writer's seq_cst pre-read acquires that store in turn, and
  so every push made before the value the trace read, along the chain of
  writers to that slot, is visible to certification's ring drains. A writer
  whose test saw marking off makes no pre-read, so the chain breaks at its
  store; the two passes of the in-flight check cover that case (barrier.md,
  "The release/acquire chain").

Callers go through `SharedGC::markCell`, as local code goes through
`GCBase::markCell`. It is static, so a verifier can walk a cell without a
heap. It is defined in `SharedGC-inline.h` because `SharedSlotVisitor.h`
includes `SharedArray.h`, which includes `SharedGC.h` (`GCBase-inline.h`
breaks the same cycle). It only forwards today; anything a visit later needs
from the heap (a stride, a shape) goes here.

---

## Marking and the mark worklist

`SharedGC::MarkAcceptor` is a private nested class, declared in `SharedGC.h`
and defined only in `SharedGC.cpp`. It holds a `SharedGC &` to reach the
private `markState_`, which has no accessor (barrier.md, "Why a barrier never
touches a destroyed `MarkState`"). Its one method,
`bool accept(SharedGCCell *cell)`, marks and queues a cell unless it is
already marked, and returns whether this call marked it.

* **Every source goes through it**: slots via `markCell`, roots via
  `markRoots`, ring entries via `drainBarrierWorklist` and `certifyRound`.
  A cell is marked when it is queued, so it is queued at most once per cycle,
  and a mark left from before the cycle would keep a reachable cell from being
  traced. That is why `beginCycle` clears every mark.
* **Only the thread running the cycle marks** (the collector thread, a test,
  or `~SharedGC`), so `accept` loads the bit and then sets it unconditionally,
  with no test-and-set. A second marking thread's bits would have no ordering
  edge to the sweep (step 7), and the worklist is a plain container. A runtime
  only pushes onto its own ring, and sets mark bits only by black allocation
  (marking.md CE-2, a runtime must not write mark bits). `setCellMarkBit` still
  uses `fetch_or`, because black allocation can set a neighbouring bit in the
  same byte. `collect()` keeps the one-thread rule by holding `cycleMtx_` for
  the whole cycle; the step-by-step entry points only assume it, and must not
  be called on a heap with a collector thread.
* **The return value is what certification counts.** A round that marks
  nothing new has found nothing. Emptiness is a weaker test: a runtime that
  keeps storing already-marked cells keeps the rings from ever emptying.
* **A born-marked cell stops here and is not traced in the cycle it is born
  in.** Its outgoing edges rest on P4-dominance (marking.md channel 9,
  heap-edge insertion): a runtime can only store a reference it has boxed, and
  a boxed cell is a root at the scan, born marked, pushed when acquired (P4),
  pushed when its key is pruned before the scan (P6), or copied into the
  barrier worklist when its runtime leaves before the scan (P12). Nothing
  enforces that a runtime stores only what it has boxed.

The mark worklist, `MarkState::localWorklist`, is a `std::stack` that only the
marking thread touches, so it needs no lock or atomics; runtimes spill into
`MarkState::barrierWorklist` instead. `drainMarkWorklist()` pops and visits
cells until the stack is empty, popping before each visit because the visit
pushes onto the same stack. It has no budget: Hades's `markLimit` bounds how
long its marker holds `gcMutex_`, and this marker holds no lock while tracing.

---

## Where marked cells come from

| Source | Path to `accept` | Status |
|---|---|---|
| A slot of a traced cell | `drainMarkWorklist`, then `markCell` | Implemented |
| A box-map key (a root) | `markRuntimeRoots`, then `markRoots` | Implemented |
| The value a store overwrites (P5) | The runtime's ring, then `MarkState::barrierWorklist` when the ring spills, or `certifyRound` when it drains the ring | Implemented |
| A cell read out of a slot (P4) | As for P5 | Implemented in `Runtime::materializeSharedSlot`, which pushes the cell it reads while marking is on, before it takes a map lock. Redundant as far as bounded models show, since the materialiser publishes its key inside its window; kept as insurance. Only tests call it. |
| A map key about to be pruned (P6) | As for P5 | Implemented in `Runtime::pruneSharedObjectBoxes`, which pushes each dead key while marking is on, before it takes the map lock. Required: a key erased before the root scan may name a cell whose only remaining edge is from a born-marked cell, which is not traced this cycle (marking.md P6). It also softens a breach of the rule that a raw `SharedGCCell *` is valid only while its box is held, without making it safe (marking.md P6). |
| The keys of a runtime that leaves while marking is on (P12) | `unregisterMutator` copies them into `MarkState::barrierWorklist`; from there as for a spilled ring entry | Implemented. Required for the same reason as P6: leaving before the root scan takes every key out of the root set at once. |
| A new cell (P8) | None: it is born marked. The mark lasts only until the next cycle starts, so a never-boxed cell (a raw `makeA` cell, used only in tests) is not protected across a cycle boundary (marking.md channel 2, shared allocation) | Implemented |

Each ring entry carries the `cycleSeq_` value its barrier's test returned, and
`drainInto` drops an entry stamped with another cycle without reading the cell,
which may already be swept. Why that loses nothing, for the store barrier, the
prune and the acquisition push, is in barrier.md, "The stamp: a push that
outlives its cycle". With the two passes of the in-flight check, a late
acquisition push names a cell that is protected anyway: as the slot's value at
the snapshot, or as a key of the runtime that stored it.

### Roots

The box maps are the root set (boxing.md, "How a runtime gets a shared pointer:
allocation and acquisition"). `markRuntimeRoots()` copies the keys of every
registered runtime with `copyRootKeys` and marks them after releasing every
lock. For each runtime in a snapshot of the registry, `copyRootKeys` takes
`barrierMtx_`, checks that the runtime is still registered and tries its map
lock `sharedObjectBoxesMtx_`, then drops `barrierMtx_` and copies the keys
(`Runtime::appendSharedRootKeysUnlocked`) under the map lock alone; a busy map
is retried after a yield. `unregisterMutator` takes the map lock after the
unlink, so a runtime is not destroyed while its map is being copied. Keys of
dead but unpruned boxes are included, which can keep a cell too long, never too
short. A runtime that leaves while marking is on copies its keys first and
appends them to `MarkState::barrierWorklist` in the critical section that
unlinks it, so a scan that no longer finds it loses nothing. The lock order is
in marking.md, "State and locks"; registration and deregistration are in
barrier.md, "Registration and teardown".

---

## Waiting before tracing

Turning marking on is one atomic increment; it does not stop a barrier that
has already read `cycleSeq_`. A barrier can test "marking is off", be
preempted, and act after marking is on, skipping the acquisition push (P4) or
the store barrier's push (P5) and leaving no record anywhere. The dangerous
case needs two runtimes: one holding a raw pointer mid-materialisation that it
did not push, and another removing the last edge to that cell without
recording it (marking.md, "Turning marking on safely", counterexample CE-1).
Hence step 2.

**The in-flight check** (marking.md P1). Each runtime increments a counter
when it enters and when it leaves an outermost barrier site, with a `seq_cst`
store on entry and a release store on exit, so an odd value means it is
inside one. Right after turning marking on, `beginCycle` reads each counter
and notes the odd ones; `pollInFlightBarriers()` says whether each of those
has changed since, and `collect` polls it before tracing. An idle runtime is
never inside a barrier, so the wait never depends on one; it lasts only as
long as the barriers already in progress, and needs no safepoint, which
Hermes does not have (no poll points in interpreter, JIT or AOT code). Each
site's window, the locking and the cost are in barrier.md, "The in-flight
check". It replaces P2, each runtime confirming from its own local GC, which
could wait forever on a runtime that never ran one (marking.md, "Rejected
alternatives").

**The second pass** (step 3, `beginSecondInFlightPass`). One pass is not
enough (marking.md P1). The first wait ends once every barrier that tested
`cycleSeq_` before marking turned on has finished, but only the collector is
sure to have seen what those barriers stored. A runtime that `beginCycle`
found outside a window can open one during that wait, see marking on, and
read a cell from a slot just before one of those stores removes the cell's
last edge. If it is descheduled before it pushes, the push can land after
certification, where its stamp no longer matches, and the cell is swept while
that runtime holds it. So the second pass makes a value-preserving
read-modify-write, `cycleSeq_.fetch_add(0, seq_cst)`, then notes every
runtime inside a window again, and `collect` polls until those windows have
closed too.

The write is the point. A barrier whose `cycleSeq_` load reads it, or a later
value (every later write to `cycleSeq_` is a read-modify-write too, so it
continues the release sequence), synchronizes with it, so it sees every store
made by a window the first pass waited for and every store a runtime made
before the first pass read its counter; it cannot read a value one of those
stores overwrote. A load or a fence would give it nothing to synchronize
with. A barrier whose load came before the write is either still inside its
window at the second look, and the second wait covers it, or has left it, and
the second pass's read of its counter makes everything it did visible. So
every window is either covered by one of the passes, and everything it did
happens before the root scan, or synchronizes with the read-modify-write.
Together with the seq_cst slot store and pre-read, this also gives the form
of marking.md P7's two-way edge that termination needs.

---

## Rejected alternatives

* **A virtual acceptor, or reusing `RootAcceptor`.** `RootAcceptor` is typed on
  `GCCell` and `HermesValue`; Hades makes only root accepts virtual, for
  embedder callbacks, which this collector does not have.
* **A public acceptor that owns the worklist, checked by an owner-thread
  assert.** The acceptor needs the private `markState_`, and the assert, on an
  object built per call, would check nothing and race on a plain
  `std::thread::id`.
* **`accept(SharedValue)`.** Decoding is the visitor's job, and a non-moving
  tracer never writes a slot back. A clearable slot would need the slot's
  address, a signature change either way.
* **Reading slots like Hades's `concurrentRead`**: a plain copy under a TSan
  suppression, and a bare read when `kConcurrentGC` is off. The tracer needs
  an acquire load of the slot itself, to pair with the store in
  `SharedValue::set`, so the slot is a `std::atomic<uint64_t>`.
* **Relaxed slot loads, ordered by fences**: an acquire fence before
  certification's drains, paired with fences in the store barrier. That would
  carry the pushes to certification, but would not order a cell's
  construction before the visitor or a runtime reads it, and acquire and
  release alone cannot give the total order marking.md P7's two-way edge
  needs. Publication needs a release store and acquire loads anyway, and with
  those the fences add nothing.
* **Letting the previous sweep clear the marks.** It misses cells born since
  that sweep and marks left by an abandoned cycle.
* **Porting the rest of Hades's tracer** (cards, evacuation, weak roots,
  finalizers). Shared cells never move, hold no weak references and never run
  a destructor.

---

## Open questions

1. **Model checking.** The two passes of the in-flight check, the
   release/acquire chain and the total order the store barrier relies on (see
   "Waiting before tracing"; marking.md P1 and P7; barrier.md, "The
   release/acquire chain") are argued on paper under the C++20 memory model and
   checked by bounded GenMC models under RC11 and IMM. The models check the
   design, not the code, and only within their bounds.
2. **The materialiser** (implemented in the decided shape). It must not hold a
   raw pointer across `BoxedSharedObject::create`, which can run a local GC: if
   it did while marking was off, another runtime could remove the cell's last
   edge before marking turned on, and the cell would be swept under a dangling
   map key. So it checks for a hit outside any window, allocates the box on a
   miss, and then re-reads the slot inside one window that loads `cycleSeq_`
   with seq_cst first, pushes before taking the map lock (a push can spill
   under `barrierMtx_`, which the root scan takes before a map lock), and
   publishes the box before the window closes (boxing.md O-B11).
3. **Immediates in `SharedValue`.** Keep `getPointerOrNull`'s single load and
   make it return null for immediates; `isPointer()` would need a second load.
4. **A safety net for termination in release builds** (marking.md, "A safety net
   for P10"). The sweep frees whatever is unmarked, so one unbarriered store is
   silently fatal, and certification still finds nothing new. Builds with
   `HERMES_SLOW_DEBUG` re-trace from the box-map keys at every `endCycle` and
   assert that everything reachable is marked. That check is one-way, because
   SATB and black allocation can leave unreachable cells marked. It walks
   only what the keys reach, never the whole heap, and checks each pointer
   against the region headers before following it. Release builds have no
   such guard.
5. **`barrierMtx_` across the root scan.** Settled: the scan holds it only to
   find each runtime and try its map lock, never across a copy (boxing.md O-B3).
6. **Teardown.** Settled: `~SharedGC` stops the collector thread and runs one
   more cycle, with no roots, whose sweep frees whatever is left (boxing.md
   O-B4).

---

## Tests

Shared-heap code builds only with `HERMES_ENABLE_SHARED_HEAP`, which is off by
default; enable it with `-DHERMES_ENABLE_SHARED_HEAP=ON` in CMake or
`-c hermes.enable_shared_heap=true` in Buck. Off compiles out
`lib/VM/SharedHeap/`, `BoxedSharedObject` and the `Runtime` glue.

* `SharedSlotVisitorTest.cpp`, `SharedMarkAcceptorTest.cpp`: the tracing core.
* `SharedRootScanTest.cpp`: the root scan.
* `SharedCertifyTest.cpp`: certification and abort.
* `SharedInFlightTest.cpp`: the in-flight check, including that the second
  pass waits for a window opened during the first wait
  (`SecondPass_WaitsForAWindowOpenedDuringTheFirst`) and does not change
  `cycleSeq_` (`SecondPass_LeavesMarkingOn`).
* `SharedCollectTest.cpp`: whole collections, including one that has to wait
  for a barrier in flight, one in which a runtime leaves during the wait, and
  one that has to wait for a window opened during the first wait
  (`Collect_WaitsForAWindowOpenedDuringTheFirstWait`).
* `SharedAttachTest.cpp`: attaching runtimes (registration, deregistration
  after shutdown), plus end-to-end collections through real runtimes,
  including a runtime that detaches, and one that prunes a key, while the
  collector waits.
* `SharedMaterializeTest.cpp`: the materialiser, including hits and misses,
  the orderings inside its window (through its test-only hook), and a cell
  kept when marking turns on inside the window or during the wait.
* `SharedCollectorThreadTest.cpp`: the collector thread's footprint,
  exhaustion and time triggers, the gap between cycles, `collect()` taking
  turns with the thread, stopping with the heap, and runtimes that store and
  materialise while it collects.
* Teardown: `Teardown_FreesCellsLeftInTheHeap` (`SharedGCTest.cpp`) and
  `Teardown_InTheLastRuntimeFreesWhatItBoxed` (`SharedAttachTest.cpp`).

`MarkCell_WalksLengthNotStride` reuses a freed slot so the slack past `end()`
holds stale pointers. Fresh slack is zero, decodes as empty and is skipped, so
it would hide a loop over the stride; ASan misses it too, because allocation
unpoisons the whole stride. The dirty recycled slot plus the recorded count is
the only guard; do not weaken it. To check the compile-time guard by hand,
remove the `SharedArray` overload of `visitSlots` and confirm that
`SharedSlotVisitor::visit` no longer compiles.

```bash
cmake -B cmake-build-asan -DHERMES_ENABLE_SHARED_HEAP=ON
cmake --build cmake-build-asan -j 10 --target HermesVMRuntimeTests
cmake-build-asan/unittests/VMRuntime/HermesVMRuntimeTests \
  --gtest_filter='*Shared*'
```
