# The SATB Store Barrier and the Cycle Boundary

Date: 2026-08-31
Revised: 2026-11-24
Status: Partly implemented; see the table in the summary.
Relates to: [marking.md](marking.md) (the protocol: rules P0-P12, the numbered channels, the lock
order, and why the collector must wait after turning marking on), [boxing.md](boxing.md) (the box
map, which is the root set, and `makeBoxed`), [allocator.md](allocator.md) (black allocation,
`mtx_`), [marking-implementation.md](marking-implementation.md) (the tracer)

---

## Summary

The shared heap's store barrier and the ring it records into, the start of a marking cycle (with the
in-flight check) and its end (certification), and the lifetime and lock rules around them.

* **SATB** (snapshot-at-the-beginning): a cycle keeps every cell that was reachable at its start
  (marking.md gives the exact snapshot point), so a store first records the value it overwrites.
  **Marking is on** while `SharedGC::cycleSeq_` is odd.
* **Black allocation**: every new cell is marked when it is allocated, whether or not marking is
  on. A cell born after a cycle starts, up to that cycle's sweep, is neither traced nor freed by
  it; the reset at the start of the next cycle clears the mark.
* **Push**: append a cell to the runtime's own ring, with a **stamp**: the `cycleSeq_` value the
  barrier's test returned. **Certification**: the collector's final check that marking is complete
  (marking.md P10).
* **The materialiser** (`Runtime::materializeSharedSlot`) reads a shared slot and returns a box for
  the cell in it, creating the box if needed (marking.md channel 1).

| Mechanism | Status |
|---|---|
| Store barrier (`SharedValue::set`, `SharedMutator::snapshotWriteBarrier`), `initField` | Implemented. Only `SharedArray::set` calls it, and no JS path reaches that yet (boxing.md O-B7). |
| Rings, spill, steal, stamps, the registry, `barrierMtx_`, `MarkState`, the cycle entry points; a `Runtime` created with a `SharedGC` registers a `SharedMutator` | Implemented |
| `SharedGC::collect()`, and the collector thread that calls it | Implemented; only tests create a `SharedGC` |
| Wait after turning marking on: the in-flight check, two passes | Implemented |
| Wait after turning marking on: confirmation from each runtime's local GC | Retired, because it could wait forever |
| Soundness of the two passes and the slot memory orders | Argued on paper under the C++20 memory model; checked by bounded GenMC models of the design, not the code (open question 1) |
| Prune push (P6); a runtime leaving while marking is on copies its keys into `barrierWorklist` (marking.md P12) | Implemented; only tests reach the prune, through a lookup, `makeBoxed` or the materialiser |
| Acquisition push (P4), in the materialiser | Implemented; only tests call it |
| `Atomics` read-modify-write (channel 10) | Not yet implemented |
| Publication ordering of shared slots (boxing.md O-B10) | Implemented: `set` stores with seq_cst, a release, and every slot read that may be followed to a cell is an acquire |
| A stale stamp matching again after 2^31 abandoned cycles | Not handled (accepted) |
| One thread runs a whole cycle, from `beginCycle` to the sweep | Enforced for `collect()`, which holds `cycleMtx_` throughout; assumed for the step-by-step entry points |

Code: `include/hermes/VM/SharedHeap/`, `lib/VM/SharedHeap/`, `BoxedSharedObject.h` and `.cpp`, and
the glue in `Runtime.h` (the box map, its lock, `sharedMutator_`) and `Runtime.cpp`, all under
`include/hermes/VM/` and `lib/VM/`. Tests: `unittests/VMRuntime/Shared*Test.cpp` and
`BoxedSharedObjectTest.cpp`. Built only with `HERMES_ENABLE_SHARED_HEAP`, which is off by default;
enable it with `-DHERMES_ENABLE_SHARED_HEAP=ON` in CMake or `-c hermes.enable_shared_heap=true` in
Buck.

---

## Shape

```
Runtime::sharedMutator_   unique_ptr<SharedMutator>; null if the runtime has no SharedGC
  ring_                   SharedBarrierRing: 256 entries {cell, seq}
  inFlight_               atomic<uint32_t>; odd while the runtime is inside a barrier window
  barrierDepth_           how many windows are open; only the runtime touches it
SharedGC
  cycleSeq_               atomic<uint32_t>; odd while marking is on
  barrierMtx_             std::mutex
  mutators_               vector<SharedMutator *>: every registered runtime
  markState_              Optional<MarkState>: exists only during a cycle
    barrierWorklist       cells taken from rings (spill, steal, unregister), and the keys of a
                          runtime that leaves while marking is on; guarded by barrierMtx_
    localWorklist         the collector's own mark stack; no lock
    awaited               runtimes in a window at either in-flight pass; written only by the
                          collector, under barrierMtx_
```

A barrier appends to its own runtime's ring without a lock. A full ring **spills**: `push` calls
`SharedGC::spill`, which moves the entries into `barrierWorklist` under `barrierMtx_`. Certification
**steals**: the collector drains every registered ring itself (marking.md P7a). Spilling hands work
over, so `barrierWorklist` has no cap, and a barrier can block only on `barrierMtx_` in `spill`.

---

## The store barrier

`SharedValue` is the slot: `raw_` is a `std::atomic<uint64_t>`, and a copy is an acquire load
(the type must stay copyable, since `SharedArray::at` returns one by value). `operator=` is deleted,
so a plain assignment cannot skip the barrier: `set` is the barriered store, and `initField` the
unbarriered one, for slots not yet published. `set(v, mutator)` opens a barrier window, runs
`mutator.snapshotWriteBarrier(*this)`, stores with seq_cst, and closes the window:

```
mutator.enterBarrier()                  // outermost: in-flight counter to odd, seq_cst
seq = gc.cycleSeq()                     // seq_cst: is marking on, and which cycle?
if (seq is odd) {                       // marking on; with it off the barrier ends here
  old = slot.getPointerOrNullSeqCst()   // the pre-read: one seq_cst load of the slot
  if (old) mutator.push(old, seq)       // ring entry {old, seq}; spills first if the ring is full
}
slot.store(v, seq_cst)                  // always; set() does this after the barrier returns
mutator.exitBarrier()                   // outermost: in-flight counter to even, release
```

* **Record, then store**, the rule `GCPointer<T>::set` also follows. The store is seq_cst even
  with marking off or nothing to push: as a release it carries this runtime's earlier pushes and,
  after a pre-read, the previous writer's, and it takes part in the single total order the barrier
  relies on (see "The release/acquire chain"). So a published slot needs `set` even when it is
  empty.
* **Reads.** Every slot read that may be followed to a cell is an acquire: a copy (so
  `SharedArray::at`), `getPointer`, and `getPointerOrNull`, which the tracer uses. The pre-read uses
  `getPointerOrNullSeqCst`, a seq_cst load, instead. `isEmpty` and `isPointer` stay relaxed, since
  nothing is read through their answer. With the store's release, a reader that follows the pointer
  sees the cell as it was built, and so does the tracer, which reads the cell's mark bit next, and
  its kind and length if it traces it; this is the publication ordering of boxing.md O-B10.
* **No mark bit is read or written** (`Store_TouchesNoMarkBit`): black allocation is a runtime's
  only mark-bit write (marking.md CE-2).
* **`initField`** is only for a slot nothing else can reach yet. A new cell's slots hold leftover
  bytes, since allocation returns uninitialised storage, so each must be written before the cell is
  published: by `initField`, or cleared to `SharedValue::empty()` as `SharedArray`'s constructor
  does. `set` on an unwritten slot would push a garbage pointer for the collector to mark through.
  A constructor also needs `initField` because allocation holds `mtx_` across it (`makeBoxed`, and
  `makeA` in tests), and a spill would take `barrierMtx_` after `mtx_`, against the lock order.
* The caller passes its `SharedMutator &`, which avoids a TLS load on every store.
* **Cost.** The seq_cst store is an `xchg` on x86, on top of the window entry's; on AArch64 it is
  the same `stlr` a release store uses; on ARMv7 it adds two `dmb`s. With marking on, the barrier
  adds the pre-read and a ring append, which need no fence on x86 or AArch64 and a `dmb` each on
  ARMv7.

### The ring

* `SharedBarrierRing` holds 256 entries `{SharedGCCell *cell; uint32_t seq;}` behind monotone
  cursors `tail_` and `head_`.
* One producer, the runtime's thread, pushes without a lock. Two consumers take turns under
  `barrierMtx_`: the producer when it spills, the collector when it steals.
* Each cursor is published with release and read with acquire by the other side, which orders the
  entries and slot reuse. The barrier's own ordering comes from the slot store, which releases the
  push to whoever reads the stored value with acquire, not from the cursors.
* `headCache_`, the producer's copy of `head_`, can only overstate how full the ring is (an early
  spill, never an overwrite), and `full()` refreshes it on its slow path. Tests must spill
  repeatedly (`KeepsSpillingAfterTheFirstSpill`, `Push_KeepsSpillingPastTheFirstSpill`): one spill
  passes whether or not that refresh works.
* `drainInto(out, seq)` appends the entries stamped `seq` and drops the rest without reading their
  cells; `discardAll()` drops them all.

---

## Starting a cycle

One thread runs a cycle: start, wait, second pass, wait again, scan roots, rounds of drain then
certify, end, sweep (marking.md, "A cycle"). Each round drains first, because `certifyRound`
requires an empty local worklist.

`beginCycle()` first creates `markState_` holding `barrierMtx_`, because `spill` tests `markState_`
under it. Then, holding `mtx_`, it clears every mark bit (`whitenAllLocked`) and turns marking on
with a seq_cst `fetch_add` that makes `cycleSeq_` odd. marking.md P8 says why the reset is needed
(`BeginCycle_ResetsMarksLeftByAllocation`) and why `mtx_` covers both parts. A spill between the two
steps finds a `MarkState` but an even `cycleSeq_`, so `drainInto` drops every entry unread. Last,
holding `barrierMtx_` again, it notes every runtime inside a barrier window
(`noteBarriersInFlight`), for the first pass of the in-flight check below.

### The in-flight check

A barrier that tested `cycleSeq_` before marking turned on can still act afterwards, unrecorded, so
the collector waits for such barriers, then makes a second pass and waits again, before it scans
roots. marking.md P1 and "Turning marking on safely" give the hazard and why the check works; this
section gives the sites, the nesting, the collector's walk, the second pass and the cost.

Each runtime's `SharedMutator` gets a counter that is odd while the runtime is inside a barrier. A
site's window opens at its outermost entry, just before its `cycleSeq()` test, and so before any
slot read it acts on, and closes after the effect the test guards. The materialiser's hit check is
not a site: it reads the slot outside any window, only as a lookup key. The sites:

| Site | Records | Window closes after | Status |
|---|---|---|---|
| Store, `SharedValue::set` (P5) | the overwritten value | the slot store | implemented |
| Materialiser (P4) | the cell it read | publishing the box; one window, after the box allocation (open question 2) | implemented |
| Prune, `Runtime::pruneSharedObjectBoxes` (P6) | each dying key, before taking the map lock | the push | implemented |
| `Atomics` read-modify-write (channel 10) | the value read before the operation | the operation | not implemented |

The store's window must include the store: with marking off the store is the barrier's whole effect,
so a window that closed at the push would cover nothing. Every site follows the same protocol, and
so must the future `Atomics` read-modify-writes and any native code that follows or stores a pointer
read from a slot: open a window, test `cycleSeq_` with seq_cst, read any slot only after that test,
and push when marking is on. Native code may use the pointer only inside that window; to keep the
cell longer it must box it as the materialiser does, allocating the box before the window opens and
publishing the key before it closes. An `Atomics` operation must be a seq_cst read-modify-write on
the slot, after a seq_cst pre-read. A window must also be short and close on every path. Nothing
in it may call JS, raise or return an exception, or wait for the collector, which waits for the
window; and nothing in it should allocate on a GC heap, or the collector waits through a local GC.
So every conversion and allocation comes first: a `ToNumber` that can call `valueOf`, a string's
migration, `makeBoxed`, the materialiser's box. A window left open keeps its runtime's counter odd,
and the first cycle to wait for it stalls in the in-flight check, holding `cycleMtx_`, so no cycle
completes until that runtime detaches; only `~SharedMutator`'s debug assert notices.

* **Nesting.** Windows may nest, so a runtime keeps a plain nesting depth, and only the outermost
  entry and exit write the counter: the entry with a seq_cst store, the exit with a release store.
  No site opens one inside another today: the materialiser prunes before its window, so the lookup
  inside it has nothing to prune.
* **The collector's walk.** After the turn-on, `beginCycle` calls `noteBarriersInFlight`, which
  holds `barrierMtx_`, reads every registered runtime's counter, and records the odd ones in
  `MarkState::awaited`; `beginSecondInFlightPass` calls it again for the second pass. Each
  `pollInFlightBarriers()` call retakes the lock, drops every recorded runtime that is no longer in
  `mutators_` (one that is gone is clear) or whose counter has changed, and says whether any is
  left; `collect` yields between calls. Nothing waits holding `barrierMtx_`, which a runtime inside
  a barrier may need in `spill`, and nothing keeps an index into `mutators_` across a release of the
  lock, since `unregisterMutator` moves the last entry into the removed slot. A runtime that
  registers after `beginCycle`'s read is ordered after the turn-on by the lock, so the first pass
  never waits for it. A runtime registered at a freed one's address is read in its place, which at
  worst waits for its window too.
* **The second pass.** Once the first wait is over, `beginSecondInFlightPass()` makes a seq_cst
  `cycleSeq_.fetch_add(0)`, which leaves the value unchanged, notes the open windows again, and
  `collect` polls until they close. The write is the point: a barrier whose `cycleSeq_` load reads
  it, or a later value (every later write to `cycleSeq_` is a read-modify-write too, so the release
  sequence goes on), synchronizes with it and sees everything the first pass learned; a load or a
  fence would give it nothing to synchronize with. A window whose load came before the write was
  still open when the second pass looked, or had closed with its effects visible to that look, so
  the second wait orders its effects before the root scan. So every window is either covered by one
  of the passes, its pushes and publishes happening before the root scan, or synchronizes with the
  read-modify-write. One pass is not enough: the first wait ends once every barrier that tested
  `cycleSeq_` before the turn-on has finished, but only the collector is sure to have seen the
  stores those barriers made. A runtime that the first pass found idle can open a materialiser's
  window during that wait, see marking on, and read a cell from a slot just before one of those
  stores removes the last edge to it; its push (P4) can then land after certification and be
  dropped by its stamp, and the cell is swept while that runtime holds it. The second pass costs
  the runtimes nothing.
* **Cost:** two stores per outermost site, even with marking off, on top of the barrier's seq_cst
  load and branch. The entry store is seq_cst: `xchg` on x86; `stlr` on AArch64, where the following
  `ldar` of `cycleSeq_` needs no `dmb`. The exit store is release, a plain store on x86 and `stlr`
  on AArch64; it only has to carry the window's effect to a collector that sees the counter move,
  because every later counter store is also at least a release. On ARMv7, which Hermes ships, the
  check takes the off path from one `dmb` to four, and the seq_cst slot store adds two more. A later
  option is an asymmetric fence on the collector's side (`membarrier` on Linux,
  `FlushProcessWriteBuffers` on Windows), making the entry a plain store too; the collector would
  then need a fence of its own after the wait, since, without same-thread release sequences
  (dropped in C++20), a poll that reads a plain entry does not synchronize with the exit before it.
  Apple platforms have no such API.

`SharedInFlightTest.cpp` covers the wait, mostly straight-line since each barrier step is a separate
call: a store that tested "marking is off" holds the wait until it stores; an inner exit does not
end an outer window; a busy runtime does not starve the wait; an idle one does not delay it; a
runtime that attaches after marking turned on is not waited for by the first pass; a runtime can
leave or spill during the wait; `set` opens exactly one window, or none when already inside one;
and the second pass leaves marking on (`SecondPass_LeavesMarkingOn`) and waits for a window opened
during the first wait (`SecondPass_WaitsForAWindowOpenedDuringTheFirst`). `SharedCollectTest.cpp`
checks that `collect` waits for a barrier in flight before tracing, waits again for a window opened
during that wait (`Collect_WaitsForAWindowOpenedDuringTheFirstWait`), and keeps the keys of a
runtime that leaves during the wait. `SharedAttachTest.cpp` checks that a cell survives when the
runtime that boxed it detaches, or prunes its key, during the wait.

---

## Ending a cycle

### Certification

`certifyRound()`:

1. Asserts that marking is on, that no window noted by either in-flight pass is still open, and
   that the local worklist is empty.
2. Marks the spilled backlog outside the lock (`drainBarrierWorklist`), keeping long work off
   `barrierMtx_`. If that marked anything, it returns false and the caller traces again.
3. Takes `barrierMtx_`, with no fence before the first ring read. The trace read every slot with an
   acquire load, which synchronizes with the seq_cst store of the writer whose value it read, so
   that writer's pushes, and those of the writers before it on that slot's chain of writers, are
   visible to the ring reads (P10 clause iii; see "The release/acquire chain"). The lock's acquire
   would not serve, since a barrier does not take the lock on its fast path.
4. Drains every ring into `barrierWorklist` and marks each entry. If that newly marked any cell, it
   returns false, and those cells wait on the local worklist.
5. Otherwise it turns marking off before releasing the lock, and returns true.

Steps 3 to 5 are one critical section, so the barrier worklist and the rings are observed together.
marking.md P10 says why that matters, and why the measure is "newly marked" rather than "empty". A
push after the turn-off has a stale stamp and is dropped (see "The stamp").

`endCycle()` then destroys `MarkState`, and the caller sweeps on the thread that traced
(allocator.md, "What publishes the marks to the sweeper"). `abortCycle()` turns marking off and
destroys `MarkState` without certifying. The caller must not sweep, because unmarked no longer
means unreachable; the next `beginCycle` clears the partial marks.

### The release/acquire chain

**The slot store is a release.** The push is sequenced before it, so a reader that loads the
stored value with acquire also sees the ring's new tail. Without that the store could become
visible before the tail: the collector traces the cell, reads the new value, steals a ring whose
tail has not moved, and certifies, losing the old value with only one runtime involved. The
tracer's acquire load (`getPointerOrNull`) is the other half, so certification needs no fence
(step 3).

**The pre-read is an acquire** (a seq_cst load), which carries that pairing past a second writer. A
release store synchronizes with an acquire load only if the load reads its value or a value in its
release sequence (C++20 [atomics.order]p2), and a release sequence continues only through
read-modify-writes ([intro.races]p5), so a second runtime's store to the slot ends it, and a
collector that reads the second runtime's value may see a stale `tail_` on the first runtime's
ring. The second writer's pre-read reads the first writer's value and so synchronizes with the
first writer's store, and the second writer's own store then releases the first writer's pushes
with its own, and so on along the chain of writers:

> **Chain lemma.** If the collector's read of a slot returned the value written by a store whose
> barrier saw marking on, then the ring publication of every push by every writer on the reads-from
> chain ending at that store happens-before the certifying round's ring reads.

The cases below rely on this form of marking.md P7's two-way edge: a barrier that sees marking on
never pre-reads a value older than one whose store came before the turn-on. It can still miss a
value another runtime stored during the cycle, but that runtime still protects the value (a key in
its map, a push, or a birth mark), which is all the cases need.

The total order of seq_cst operations provides this. Every `set` store is seq_cst, and so are the
turn-on and the barrier's test and pre-read; the test reads the turn-on or a later value, so in
their single total order a store before the turn-on also comes before the pre-read, which therefore
returns that store's value or a later one. With release and acquire alone it could return an older
one. A runtime stops protecting a cell only by pruning its key or by detaching. The prune tests
`cycleSeq_` with seq_cst first, and so does `unregisterMutator` when it finds a `MarkState`: if the
test saw marking off, every slot store the runtime made of that cell came before the turn-on; if it
saw marking on, the runtime pushes the cell or copies its key. A detach before `beginCycle` creates
the `MarkState` makes no test, but `barrierMtx_`, which `beginCycle` takes first, orders it before
the turn-on.

For a value `V` overwritten by a store whose barrier saw marking on:

1. **Case 1: the collector read `V`, or a value stored before `V`.** If it read `V`, it marked `V`
   while tracing. If it read an earlier value, `V` was stored during the cycle, since after the
   wait every store whose barrier tested `cycleSeq_` before the turn-on happens-before the trace. So
   `V` is covered like any reference stored during the cycle (marking.md channel 9).
2. **Case 2: the collector read the overwriting value, or a later value.** If `V` was stored after
   the turn-on, it is covered as in case 1. Otherwise no pre-read that sees marking on misses `V`,
   so the reads-from chain back from the value read reaches a barrier that pushed `V`: the
   overwriting one, or a later one whose pre-read returned `V`. By the chain lemma the collector
   sees that push. If `V` is unmarked, certification marks it and the round fails, so `V` is traced
   next round (`Barrier_RecordedButUnmarked_FailsTheRound`). The chain may instead stop at a late
   store whose barrier saw marking off, which does no pre-read; the two passes cover that. The late
   store's window was open at the turn-on, so the first pass orders the store before the second
   pass's read-modify-write. The overwriting barrier's test therefore read `cycleSeq_` before that
   write (had it read the write, the late store would happen-before the overwriting store, which
   precedes it on the slot), so the second pass covered its window, and its push of `V`
   happens-before the trace.
3. **Case 3: the collector never read the slot.** The containing cell was not live at the snapshot,
   so the edge is on no live path; or the cell was born this cycle and is not traced, and what it
   holds is covered by P4-dominance (marking.md channel 9), which also relies on the prune push
   (marking.md P6) and on a leaving runtime copying its keys (marking.md P12), not by this barrier.

There is no fourth case, so a round cannot certify and then find new work.

The chain needs every slot access to be an atomic operation on the slot: a plain read under a
sanitizer suppression gives the release nothing to pair with. Hence the atomic `raw_` and the
tracer's single acquire load per slot (`getPointerOrNull`). The chain alone needs only release and
acquire; the store, the pre-read and the barrier's `cycleSeq_` load are seq_cst for the total
order above, the load also for the in-flight check, and nothing needs `membarrier`.

The chain covers only pushes followed by a slot store (P5). The prune push (P6) is ordered by the
map lock instead, and the keys a leaving runtime appends (marking.md P12) by `barrierMtx_`, under
which the root scan checks that a runtime is still registered. The acquisition push (P4) rests on
marking.md's provenance rules, which are relied on and not enforced: in production a runtime names a
cell only by allocating it with `makeBoxed` or by materialising it, and uses a raw pointer only
while it holds the cell's box.

### The stamp: a push that outlives its cycle

A barrier can read `cycleSeq_` in one cycle and push after that cycle has ended, for example after
waiting on `barrierMtx_` in `spill`. The cell may be swept by then, and if it was the last cell in
its region, `releaseRegion` may have unmapped the segment. Marking that entry would mask a freed
pointer to a segment header and write unmapped memory, which no release-build check catches.

So the barrier's single `cycleSeq()` load says both whether marking is on and which cycle it is
(two atomics could be read on either side of a change), and the entry carries that value as its
stamp. A drain keeps only entries whose stamp equals the `cycleSeq_` value read under `barrierMtx_`,
comparing integers without reading the cell (`Store_AfterTheCycleEnded_IsDroppedNotAccepted`).
`accept`, the only code that consumes a drained entry (and the code that would mask it to a segment
header), never sees a dropped one. With no `MarkState`, `spill` calls `discardAll()`
(`Spill_WithNoCycleDiscardsTheRing`); between certification and `endCycle`, `cycleSeq_` is even, so
every entry is dropped.

Dropping a stale store-barrier entry (P5) is sound by the cases in "The release/acquire chain": if
a cycle's certifying round missed the push, its trace also missed the store that followed, so it
read the old value itself or did not need it. A prune push (P6) that a root scan needed is never
late, because the map lock orders it before the scan's copy of that map. A prune that tested before
marking turned on can erase with no fresh record. Another runtime may still reach the cell, through
its own key or through a slot, but every slot store this runtime made of the cell came before that
test, so a barrier that sees marking on and overwrites one pre-reads the cell and pushes it (the
total order in "The release/acquire chain"). A late acquisition push (P4) may be dropped too: with
the two passes, the cell is still protected by the slot's snapshot (the trace finds it there, or a
barrier records it when overwriting it) or by the key of the runtime that stored it there.

`cycleSeq_` wraps. A certified cycle drains every ring at least once (by a steal or by
`unregisterMutator`). An entry pushed after that cycle's last drain, by a barrier still in its
window, stays in the ring until the next certified cycle drains it and its stamp drops it, so no
entry outlives two certified cycles. An abandoned cycle drains nothing, so an idle runtime's entry
could survive 2^31 abandoned cycles and match again; if abandoning becomes routine, `abortCycle`
should discard the rings.

---

## Lifetimes and locks

### Why a barrier never touches a destroyed `MarkState`

`MarkState` exists only during a cycle, while barriers on many threads keep running.

1. A barrier reaches `cycleSeq()`, its slot, its own `SharedMutator` (ring, in-flight counter and
   depth), and on overflow `spill()`. Only `spill()` touches `MarkState`.
2. `spill()` takes `barrierMtx_` and tests `markState_` before using it, as `unregisterMutator`
   does. `beginCycle`, `endCycle` and `abortCycle` create and destroy `markState_` under that lock,
   and `markState_` has no accessor.
3. Everything else outlives the barrier. `cycleSeq_` and `barrierMtx_` are `SharedGC` members, and
   the `SharedGC` outlives every registered runtime: each `Runtime` holds a `shared_ptr<SharedGC>`,
   and `~SharedGC` asserts that `mutators_` is empty. The ring belongs to the barrier's own
   `SharedMutator`, and one thread cannot be inside `push` and `~SharedMutator` at once (a debug
   `detached_` flag catches a push during destruction). Shared cells never move.

So a barrier that reads `cycleSeq_` as odd and sleeps through the end of the cycle touches only its
own ring and the shared heap when it resumes; if the ring is full it spills, finds no `MarkState`,
and discards the ring (`Cycle_EndsWhileMutatorsAreStillPushing`, with threads).

This breaks if `markState_` gets an accessor, if `barrierMtx_` moves into `MarkState` (a mutex
cannot guard the lifetime of the object that holds it), if `cycleSeq_` or a ring moves into
`MarkState`, or if anything touches `barrierWorklist` without `barrierMtx_`. It also assumes one
thread runs the whole cycle. `collect()` does, holding `cycleMtx_` throughout so that the collector
thread and other callers take turns; nothing enforces it for the step-by-step entry points, which
must not be used on a heap that has a collector thread.

### `barrierMtx_` and lock order

`barrierMtx_` is a raw `std::mutex`, not `GCConcurrency.h`'s `Mutex`, which becomes `FakeMutex`
whenever `kConcurrentGC` is false (every 32-bit build, for example), while the shared heap is used
from several threads in every build. It guards `mutators_`, whether `markState_` exists,
`barrierWorklist`, and every ring drain. One lock covers all of these because certification and
`unregisterMutator` must each see the registry, the worklist and the rings held still at once
(marking.md P10 and P12). A runtime registers only once, so a second lock would gain nothing.

Only the thread running the cycle writes `cycleSeq_`: the collector thread, another caller of
`collect()` such as `~SharedGC`, or a test driving the steps one by one. Marking turns on under
`mtx_` (`beginCycle`) and off under `barrierMtx_` (`certifyRound`, `abortCycle`, and `endCycle` if
marking is somehow still on), so no spill lands between the last drain and the turn-off.

The lock order is in marking.md, "State and locks". What it means for the barrier:

* `barrierMtx_` before `mtx_`: a shared cell's constructor runs under `mtx_`, so it uses
  `initField`, never `set`.
* `barrierMtx_` before a runtime's map lock (`markRuntimeRoots`, which only tries it), never after:
  the prune push (P6), which can spill, comes before the prune takes the map lock, and is skipped
  when `sharedMutator_` is null.
* The collector never takes a runtime's `gcMutex_`. Finalizers never touch shared slots (marking.md,
  "State and locks"), so no barrier runs under `gcMutex_` or on a Hades background thread. The
  reverse does happen: a materialiser's window can take its runtime's `gcMutex_`, since the
  barriered `WeakRoot::get` in its lookup may flush Hades's barrier chunk and pause the background
  marker (`HadesGC::handleFullBarrierChunk`). That waits on nothing shared, so it can lengthen the
  collector's wait but cannot deadlock it.

The common store path never takes the lock: only `spill` does, at most once per 256 records, and it
may then wait for one of the collector's critical sections, or for a leaving runtime's. None of them
walks a box map under it: the root scan (`markRuntimeRoots`) holds it only to find each runtime and
try its map lock, and a runtime that leaves copies its own keys before taking it, so what a spill
can wait for there is the append of that copy to the barrier worklist, a memcpy made only while
marking is on (open question 4).

### Registration and teardown

* Registration and deregistration are marking.md P0 and P12: the `Runtime` constructor registers
  last, by creating the `SharedMutator`, and a post-shutdown deleter deregisters by resetting it.
  Deleters run in registration order, so one added after the constructor runs after deregistration
  and must not store into shared slots.
* `unregisterMutator` first copies the runtime's map keys, taking no lock. Then, in one critical
  section, it drains the ring into `barrierWorklist` (if a cycle exists), appends the keys to it (if
  marking is on), and removes the runtime from `mutators_`. So the collector never steals from a
  ring that is being destroyed, and it finds either the runtime, in the root scan, or its keys, in
  the worklist that certification drains (marking.md P12, channel 12). Last, it takes the runtime's
  map lock and releases it, which waits for a root scan still copying the map. The appended keys
  carry no stamp: they go straight into `barrierWorklist`, and only while marking is on, so
  certification consumes them before marking turns off; keys appended after certification would
  leave the worklist non-empty for `endCycle`.

---

## Rejected alternatives

Rejected designs for the wait itself (each runtime confirming from its local GC, a 0/1 flag, a
counter checked at termination, holding `mtx_` across the materialiser) are in marking.md.

* **Mutate-then-check** (store, then test whether marking is on, and push if so): `cycleSeq_` is a
  parity counter, so the later test may see marking off again (no record; the cell is swept) or a
  later cycle (a swept cell gets a current stamp, defeating the stale-entry check). A record of an
  overwritten value made after the store is visible cannot coexist with a certification that ends
  on "nothing new". Recording first is what makes a late push redundant and a late entry stale
  rather than forged.
* **A lock on every store**: Hades's one mutator can use `markState_` without a lock because only it
  destroys `markState_`. N runtimes have no single owner, and a lock on every store would serialise
  them.
* **A mutator that runs marking when the barrier worklist overflows** (Hades's
  `handleFullBarrierChunk`): a mutator blocked on a collector that is waiting for it deadlocks
  (marking.md P7a).
* **Discarding leftover ring entries when a cycle starts**: not needed, since the stamp drops them
  when they are drained. If one is ever added, it must run before marking turns on, or it throws
  away real pushes made after it.
* **Acquire and release fences around the push, with relaxed slot accesses** (an acquire fence
  after the pre-read, a release fence before the store, and an acquire fence in certification): they
  give the chain lemma, but they run only while marking is on, so they publish no new cell while it
  is off; and release and acquire alone let a pre-read return a value older than one stored before
  the turn-on, which the total order rules out (see "The release/acquire chain"). With marking on
  they also cost two fences per barrier.
* **The fences above, with two plain passes** (no read-modify-write between them): under the
  standard compilation mappings the passes appear correct on x86-TSO, ARMv8 and `dmb`-based
  ARMv7, but the argument at the C++ level needs both the seq_cst slot accesses and the
  read-modify-write, so the code does not rely on the hardware.

---

## Open questions

1. **Model checking.** The two passes, the release/acquire chain and the total-order argument are
   worked out on paper under the C++20 memory model and checked by bounded GenMC models under RC11
   and IMM, which find no lost cell; variants that drop a needed ordering, a pass or the
   read-modify-write lose one, among them a store barrier whose slot store and pre-read are only
   release and acquire. The models track one or two cells, three or four runtimes acting once and at
   most two cycles, and check the design, not the code.
2. **The materialiser** (P4; implemented in the decided shape; only tests call it). It checks for a
   hit outside any window and, on a miss, allocates the box; then, in one window entered with a
   seq_cst store, it loads `cycleSeq_` with seq_cst before re-reading the slot, pushes before taking
   the map lock (a push can spill under `barrierMtx_`, which the root scan takes before a map lock),
   and publishes the box before the window closes (boxing.md O-B11). Bounded models find the push
   redundant with the two passes; it is kept as insurance.
3. **Provenance is not enforced.** The acquisition push, and dropping a late one, rely on
   marking.md's provenance rules: a runtime names a cell only through `makeBoxed` or the
   materialiser, and uses a raw pointer only while it holds the cell's box. Nothing checks either
   (marking.md channels 4 and 5).
4. **`barrierMtx_` across the root scan.** Settled: the scan holds it only to find each runtime and
   try its map lock, never across a copy, and a busy map is retried after a yield (boxing.md O-B3).
5. **`~SharedGC` and a non-empty heap.** Settled: it runs one more cycle, with no roots, which frees
   what is left (boxing.md O-B4).
6. **64-bit atomics on 32-bit targets** (boxing.md O-B1). If `std::atomic<uint64_t>` is not
   lock-free there, slot accesses stay correct but each takes a lock.
7. **The barrier worklist's growth.** `barrierWorklist` has no cap, and `collect()` empties it only
   in `certifyRound`. Every entry a runtime pushes while marking is on lands there when its ring
   spills, repeats of already-marked cells included, so a runtime that stores in a tight loop during
   a long trace grows it for as long as the trace runs, in memory the heap's budget does not count.
   Hades caps its barrier worklist by having the mutator drain it, which is rejected here. Open:
   whether to accept the growth or bound it, for example by draining the worklist between batches of
   tracing, or by dropping at the spill the entries whose cell is already marked.
