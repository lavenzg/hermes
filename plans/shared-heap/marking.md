# Marking the Shared Heap

Date: 2026-08-24
Revised: 2026-11-24
Status: Partly implemented; see the rule table in the summary.
Relates to: [boxing.md](boxing.md) (boxes and the per-runtime map), [barrier.md](barrier.md) (the
store barrier, its rings, and the details of the in-flight check), [allocator.md](allocator.md)
(segments, mark bits, sweep), [marking-implementation.md](marking-implementation.md) (the tracer)

---

## Summary

How the shared heap's collector marks: what the roots are, what each barrier records, how a cycle
starts and ends, and why. Rules are labelled P0-P12 plus P7a, and the ways a cell becomes reachable
or unreachable are numbered channels 1-12. The other shared-heap documents cite both, so
the numbers are stable; a retired rule keeps its number.

* **SATB** (snapshot-at-the-beginning): a cycle keeps every cell that was reachable at its snapshot.
  The snapshot is taken when the first pass of the wait that follows turning marking on ends (P1):
  by then every store whose test came before the turn-on has landed, and every window opened
  afterwards, until certification, sees marking on. To keep that true while runtimes run, a store
  records the value it overwrites, not the value it writes.
* **Black allocation**: every new cell is born marked and is not traced in the cycle it is born in.
  Each cycle starts by clearing every mark, so during a cycle a cell is marked only if the collector
  marked it or it was born in that cycle.
* **P4-dominance**: every reference a runtime can store into a shared cell was allocated (born
  marked), acquired with a push (P4), or is a key in its box map, which the root scan copies, the
  prune pushes before erasing (P6), or `unregisterMutator` copies if the runtime leaves before the
  scan (P12). It is what covers the contents of a cell born this cycle, which the collector does not
  trace.
* **Marking is on** while `SharedGC::cycleSeq_` is odd. **Push**: append a cell, stamped with the
  cycle, to the runtime's own ring.
* **Barrier site**: code that tests whether marking is on and acts on the answer. Its **window**
  runs from just before the test through the effect the test guards. A runtime is **inside a
  barrier** while it is inside a window.
* **The materialiser**: the code that reads a shared slot and returns this runtime's box for the
  cell in it, creating and publishing the box if needed.

Built only with `HERMES_ENABLE_SHARED_HEAP`, which is off by default (enable it with
`-DHERMES_ENABLE_SHARED_HEAP=ON` in CMake or `-c hermes.enable_shared_heap=true` in Buck); off
compiles out `lib/VM/SharedHeap/`, `BoxedSharedObject` and the Runtime glue. Only unit tests create
a `SharedGC` or attach a runtime to one; no JS feature or embedder API reaches the shared heap yet.
`SharedGC::collect()` runs a whole cycle. A `SharedGC` created with a `CollectorConfig` runs it on a
collector thread, when its footprint, a failed allocation or the time since the last cycle calls for
one (allocator.md, "Budget, mappings and exhaustion").

| Rule | Covers | Status |
|---|---|---|
| P0 | Membership | implemented |
| P1 | Turn marking on, then wait | implemented: the in-flight check, two passes |
| P2 | Each runtime's local GC confirms that marking is on | retired; replaced by P1's in-flight check |
| P3 | PARKED state | retired; never built |
| P4 | Acquisition push | implemented; only tests call the materialiser |
| P5 | Store barrier | implemented |
| P6 | Push before the prune erases | implemented |
| P7 | Reading the roots | implemented |
| P7a | Rings, spill, steal, stamps | implemented |
| P8 | Black allocation and the reset | implemented |
| P9 | Atomic mark bitmap | implemented |
| P10 | Termination | implemented |
| P11 | Materialiser contract | implemented; only tests call the materialiser |
| P12 | Detach | implemented |
| channels 10, 11 | `Atomics` read-modify-writes, `SharedArray` indexed access from JS | not yet implemented |

A cycle may run while attached runtimes allocate (P8), acquire (P4, P11), store (P5), prune (P6) and
detach (P12). The argument for the two-pass wait (P1), P7's two-way edge and the materialiser's
shape is made on paper and checked by bounded GenMC models of the design, not the code (Open
questions: Model checking). Only tests create a `SharedGC`, so only tests run cycles: step by step,
through `collect()`, through the collector thread, or through `~SharedGC`, which runs one last cycle
to free what is left.

---

## Roots

**Map entries are the root set.** Each runtime's `sharedObjectBoxes_` maps every shared cell the
runtime references to its `BoxedSharedObject`, which the map holds weakly. A runtime gets a
`SharedGCCell *` in only two ways, and both box the cell:

* **Allocation** (implemented): `SharedGC::makeBoxed` allocates the box, then, holding the
  allocation lock `mtx_`, allocates the cell (born marked), binds the box and publishes it into the
  map.
* **Acquisition** (implemented; only tests call it): the materialiser,
  `Runtime::materializeSharedSlot`, reads a slot and boxes the cell (P4, P11).

Either way the box is rooted and in the map before VM code sees the cell. (`SharedGC::makeA` and
`SharedArray::create` skip boxing, and `BoxedSharedObject::create(runtime, cell)` binds a box
without publishing it; all three exist only in unit-test builds.) A raw pointer handed over by
another runtime (channel 5) must be boxed by the receiver the same way, which no API supports yet
(boxing.md O-B15). The handoff is legal only while the sender keeps its box alive, so the cell
stays a key in the sender's map until the receiver has boxed it.

**The raw-pointer rule.** A raw `SharedGCCell *` is valid only while a reference to its box is
held: in C++, a `PinnedValue<BoxedSharedObject>` in registered `Locals`, unless another strong root
keeps the box alive. The map's weak reference does not count. This covers values returned by slot
reads and atomic operations too, with one exception (P4): native code may use a pointer it reads
from a slot, unboxed, inside the window that read it, after that window's seq_cst test, provided it
pushes the pointer if marking is on. The rule is weaker than the `Handle` rule, since a shared cell
never moves: freeing it takes its box dying, its key being pruned and a later cycle finishing. The
compiler does not enforce it (channels 4 and 5).

**Provenance** (it relies on the two ways above and on the raw-pointer rule, and nothing enforces
either). After the wait that follows turning marking on (P1), every cell a runtime names is a key in
its map, or is held by a materialiser that saw marking on and will push it (P4). The collector
copies every key still present when it reaches that runtime (P7); a key erased before then is pushed
first (P6), and a runtime that leaves before then copies all its keys into the barrier worklist as
it unregisters (P12). Anything named later was allocated (born marked) or acquired. An acquired cell
is pushed while marking is on, and is also still protected by its slot or by the runtime that stored
it there, which covers a push that lands after certification and is dropped (P4). So every cell any
runtime can name during a cycle is marked by the time that cycle's sweep runs.

---

## Design

### State and locks

* **`cycleSeq_`** (`SharedGC`, a `std::atomic<uint32_t>`): odd exactly while marking is on;
  `markingOn()` tests that. One seq_cst load gives a barrier both the answer and the cycle; two
  atomics could be read on either side of a change.
* **Barriers** test `cycleSeq_` and push nothing while marking is off: the store barrier (P5) and
  the prune push (P6) and the acquisition push (P4). Each site also writes its runtime's in-flight
  counter on entry and exit (P1). Allocation (P8) reads no marking state; it marks every new cell.
* **Per runtime:** the map and its lock `sharedObjectBoxesMtx_`; a `SharedMutator` that owns the
  runtime's ring (`SharedBarrierRing`) and its in-flight counter (P1).
* **Per cycle:** `SharedGC::markState_`, alive from `beginCycle` to `endCycle` or `abortCycle`. It
  holds the collector's local mark worklist, the barrier worklist that rings spill into and that a
  runtime leaving while marking is on copies its keys into, and the runtimes the in-flight check is
  waiting for.

| Lock | Guards |
|---|---|
| `cycleMtx_` | one whole `collect()`, so cycles take turns, the collector thread's among them |
| `mtx_` | allocator state; held across a new cell's constructor, `makeBoxed`'s publish, and `beginCycle`'s reset and turn-on |
| `barrierMtx_` | the registry `mutators_`, whether `markState_` exists, the barrier worklist, every ring drain, and turning marking off |
| `sharedObjectBoxesMtx_` (per runtime) | the map's structure: insert, prune, and the root scan's key copy, which `unregisterMutator` waits for by taking it after the unlink |
| `collectorMtx_` | the collector thread's sleep and its stop flag; taken to wake the thread, and never held while it collects |

Order: `cycleMtx_` before every other lock; `barrierMtx_` before `mtx_` (nothing holds both); `mtx_`
before a map lock (`makeBoxed`); `barrierMtx_` before a map lock (the root scan, which only tries
it, and then drops `barrierMtx_` for the copy); never a map lock first; `collectorMtx_` last, taken
under `mtx_` or `barrierMtx_` to wake the collector thread, with nothing taken under it. Because no
map lock may be held first, the prune pushes before it takes the map lock (P6). Because
`barrierMtx_` comes before `mtx_`, a shared cell's constructor, which runs under `mtx_`, uses
`initField`, never `set`, which can spill.

* Never hold `barrierMtx_` while waiting on a runtime: that runtime may need it to spill, and one
  stalled runtime would freeze the rest (`SamplerPosix`'s mistake with `profilerLock_`). The root
  scan keeps this rule by retrying a busy map after a yield instead of waiting for it.
* The collector never takes a runtime's `gcMutex_` and never runs on a Hades background thread.
  `mtx_` never nests with `gcMutex_`, in either order. The weak-root walk runs under `gcMutex_` and
  must not allocate in the shared heap, take a shared-heap lock or push. No shared-heap lock is held
  across a local allocation. A window can take `gcMutex_`, though: the materialiser's lookup inside
  its window reads a hit through `WeakRoot::get`, whose read barrier can fill Hades's barrier chunk,
  and `handleFullBarrierChunk` can then pause the background thread under `gcMutex_`. So nothing
  that holds `gcMutex_` may wait for the shared collector, which waits for that window.
* **Finalizers never touch shared slots.** The local GC never visits shared cells, so a finalizer
  has no reason to. This keeps every barrier on its runtime's own thread: Hades can run finalizers
  on its background thread while it sweeps the old generation, and a barrier there would push onto
  the runtime's ring from a second thread, breaking the ring's one-producer rule (P7a).
* Use raw `std::mutex` and `std::atomic`: `GCConcurrency.h`'s `Mutex` and `AtomicIfConcurrentGC`
  become single-threaded stand-ins when `kConcurrentGC` is false (every 32-bit build, for one), but
  the shared heap is always used from several threads.

### A cycle

`SharedGC::collect` runs these steps on one thread:

1. **Start** (`beginCycle`): create `markState_` under `barrierMtx_` and release it; then, under
   `mtx_`, clear every mark bit and turn marking on, so no allocation lands between the two (P8);
   then, under `barrierMtx_`, note every runtime whose in-flight counter is odd (P1).
2. **Wait**, in two passes (P1). `collect` calls `pollInFlightBarriers()` until it returns true,
   yielding in between, so no runtime is still inside a barrier that tested `cycleSeq_` before
   marking turned on. Then `beginSecondInFlightPass` makes a value-preserving seq_cst
   read-modify-write on `cycleSeq_` and notes every runtime inside a window again, and `collect`
   polls until those windows have closed too. The entry points that mark, trace or certify assert,
   in debug builds, that no noted window is still open; nothing checks that the second pass ran.
3. **Scan roots** (`markRuntimeRoots`, P7).
4. **Trace and certify**: drain the local worklist and call `certifyRound`, until it certifies and
   turns marking off (P10). Shared cells never point into a local heap, so tracing never needs a
   local GC.
5. **End** (`endCycle`), then **sweep** on the same thread.

`abortCycle` turns marking off and drops both worklists. The caller must not sweep, since the marks
are partial; the next reset clears them. A sweep there frees nothing, as does one after `endCycle`
on a cycle that never certified; debug builds assert.

### Turning marking on safely

**The hazard.** Turning marking on is one atomic increment; it does not stop a barrier that has
already read `cycleSeq_`. A barrier can test "off", be preempted, and act after marking is on,
skipping its push (P4 or P5) and leaving no record anywhere. The dangerous case needs two runtimes
(counterexample CE-1):

```
X.f = Y, X is a root, and nothing else refers to Y.

u1  B's materialiser, its box already allocated, opens its window. Its test says off; it
    reads X.f into a raw Y and pushes nothing.
u2  The collector turns marking on, scans the roots and starts tracing.
u3  C stores X.f = null. C tested before u2, saw off, and records nothing.
u4  The collector traces X, reads null, never finds Y, and sweeps it.
u5  B publishes its box for Y, over freed memory.
```

One runtime alone cannot do this: its materialiser finishes before its next store, so by then Y is
a map key. It takes one runtime holding the only reference invisibly (raw, not pushed) while another
removes the only visible one without a record, and no later check can see a runtime between a raw
load and its push. A barrier still running from an earlier cycle is the same: its push has that
cycle's stamp and will be dropped, so its store must also land before tracing. Hades starts a cycle
in one uninterrupted block with its only mutator stopped. N independent runtimes have no such
moment, and Hermes has no safepoint (no poll points in interpreter, JIT or AOT code). So the
collector waits.

**The in-flight check** (implemented; barrier.md's "The in-flight check" lists the sites, the
nesting and the instruction cost). Each runtime has a counter that only it writes. A barrier site's
window runs from its outermost entry, just before its `cycleSeq()` test, through the effect the test
guards: the slot store for the store barrier, publishing the box for the materialiser, the push for
the prune, the operation for an `Atomics` read-modify-write. The runtime increments the counter at
its outermost entry, with a seq_cst store, and again at its outermost exit, with a release store
(sites nest), so odd means "inside". After turning marking on, the collector reads each counter
once: even means clear; odd means wait until the value changes. It makes that pass twice, the second
after a read-modify-write on `cycleSeq_` (see "The second pass" below), and then scans the roots.

* **Why it works.** The barrier writes its counter, then reads `cycleSeq_`; the collector writes
  `cycleSeq_`, then reads the counter. With all four seq_cst, either the barrier sees marking on, or
  the collector's read comes after the barrier's write: it then finds the window open and waits, or
  finds it closed with its effect already visible. With only release and acquire both can miss, so
  the barrier's load must stay seq_cst. The exit store, a release, is what makes the effect visible.
* **What a window must cover.** The store barrier's must include the store: one that ended at the
  push would be empty with marking off, and the unrecorded store would land outside it. The
  materialiser must read the pointer it publishes inside a window, after that window's test, and
  publish it before the window closes (P11). Its hit check needs no window: it only looks the value
  up and never follows it. Allocation needs no window: `makeBoxed` publishes under `mtx_`, which
  `beginCycle` also holds.
* **Once clear, stays clear**: every later barrier of that runtime sees marking on. So runtimes can
  be checked one at a time, and waiting for a change rather than for even cannot starve. The walk
  over `mutators_` must allow registration and unregistration while `barrierMtx_` is released
  (barrier.md, "The in-flight check").
* **Idle runtimes** are never inside a barrier, so the collector never waits for them, and P3's
  PARKED state is not needed. A wait lasts one barrier site, unless the runtime is descheduled
  inside it. No window spans a local GC: the materialiser allocates its box before it opens its
  window.
* **The second pass** (implemented). One pass is not enough: it orders each runtime's earlier
  writes, those of the windows it waited for included, before the collector, but not before other
  runtimes' later barriers. A runtime found clear can open a window during the first wait, see
  marking on, and read a cell from a slot just before a window that tested marking off, and that
  the first pass waits for, removes the cell's last edge without a record. Nothing waits for the
  reader's window, so its push can land after certification, where its stamp no longer matches,
  and the cell is swept while that runtime holds it. So `beginSecondInFlightPass` makes a
  value-preserving `cycleSeq_.fetch_add(0, seq_cst)`, then notes every runtime inside a window
  again, and the collector waits for those windows too. The write is the point: a barrier whose
  `cycleSeq_` load reads it, or a later value, synchronizes with it (every write to `cycleSeq_` is
  a read-modify-write, so a later value continues its release sequence) and so sees everything the
  first pass learned; a load or a fence alone would give it nothing to synchronize with. A window
  whose test came before the read-modify-write is still open when the second pass looks, and is
  waited for, or has closed, and that look sees its effects. So every window either has its pushes
  and publishes happen before the root scan, or synchronizes with the read-modify-write.

### Pushes that outlive their cycle

Implemented. A barrier can test in one cycle and push after that cycle has certified, and an entry
can sit in a ring into later cycles (`abortCycle` drains nothing). By then the cell may be swept and
its segment unmapped by `releaseRegion`, so accepting the entry would set a bit through a freed
pointer. So each entry carries the `cycleSeq_` value its test returned, and `drainInto` drops any
entry not stamped with the current cycle, without reading the cell. A push waits in the ring for a
spill or a steal; one made after certification's last drain, by a barrier still in its window, waits
until the next certified cycle drains it and drops it by its stamp. barrier.md, "The stamp: a push
that outlives its cycle", explains why dropping a late push is safe for the store barrier, the prune
and the acquisition push, and what happens when `cycleSeq_` wraps. A late acquisition push comes
from a window that opened after the second pass looked, so the cell it names is already protected by
the slot's snapshot or by the key of the runtime that stored it.

### The rules

**P0 — Membership** (implemented). The registry is `SharedGC::mutators_`, under `barrierMtx_`. A
runtime joins by passing a `SharedGC` to `Runtime::create`. The constructor registers last, by
creating its `SharedMutator`: after internal JS, which cannot reach shared cells, and before any
user JS. A runtime that registers mid-cycle needs no special case: if it registers after
`beginCycle` read the counters, `barrierMtx_` orders its registration after the turn-on, so every
barrier it runs sees marking on.

**P1 — Turn marking on, then wait until no runtime is inside a barrier** (implemented). One seq_cst
increment of `cycleSeq_`, then the in-flight check described under "Turning marking on safely", in
two passes: `beginCycle` notes every runtime inside a barrier, and `pollInFlightBarriers()` returns
true once each has left it; then `beginSecondInFlightPass` makes a value-preserving seq_cst
read-modify-write on `cycleSeq_` and notes every runtime inside a barrier again, and the collector
polls until each of those has left it too.

**P2 — Each runtime's local GC confirms that marking is on** (retired; replaced by P1's in-flight
check). `Runtime::markWeakRoots` would record the odd `cycleSeq_` value it read, and the collector
would wait until every runtime had. One that never runs a local GC would stall it forever; see
Rejected alternatives.

**P3 — PARKED state** (retired; never built). A runtime would have published it before blocking or
going idle, so the wait would not stall on it. Replaced by P1's in-flight check, under which an idle
runtime is never inside a barrier.

**P4 — Acquisition push** (implemented). When the materialiser sees marking on, it pushes the cell
it read, inside the window that covers the read. A map hit needs no push: the key was scanned, or
was pushed or born marked when inserted. With the two-pass wait (P1), the push is redundant as far
as bounded GenMC models can show: the materialiser publishes its key inside its window, so a window
whose test came before the second pass's read-modify-write is waited for or found closed, and one
whose test came after cannot read a cell that an unrecorded store removed. The cell it does read is
still protected by its slot (the trace finds it there, or the next store's pre-read pushes it, P5)
or by the runtime that stored it there (its key, P6, P12 or a birth mark), so the redundancy also
rests on P5, P6 and P12. The push is kept as insurance. The materialiser holds no raw pointer across
`BoxedSharedObject::create`, which can run a local GC: it allocates the box first, and reads the
pointer it publishes only inside the window that follows (boxing.md, "Acquisition: the
materialiser"). Native code that follows or stores a pointer read from a slot must read it inside a
window, after the window's seq_cst test, push it if marking is on, and use it only inside that
window. To keep the cell past that window it must box it as the materialiser does, allocating the
box before the window opens and publishing its key before the window closes. A raw pointer kept past
its window loses the cell even with the push, and copying the pointer into a cell born this cycle
without boxing it is safe only because of the push. A push that lands after its cycle has certified
is dropped by its stamp, which is safe ("Pushes that outlive their cycle").

**P5 — Store barrier: record the old value before the store** (implemented).

```
SharedValue::set(v, mutator):
    mutator.enterBarrier()              // outermost: counter to odd, seq_cst
    seq = cycleSeq_.load(seq_cst)       // one load: is marking on, and which cycle?
    if seq is odd:
        old = getPointerOrNullSeqCst()  // the pre-read: one seq_cst load of the slot
        if old: ring.push(old, seq)     // a full ring spills under barrierMtx_
    raw_.store(v, seq_cst)              // a release, and in the single total order
    mutator.exitBarrier()               // outermost: counter to even, release
```

The slot store is seq_cst whether or not marking is on, even with nothing to push, and the pre-read
is seq_cst too (see "Why the slot store and pre-read are seq_cst"). As a release, the store also
publishes a newly built cell to any reader that loads the pointer with acquire (boxing.md O-B10).
The slot is a `std::atomic<uint64_t>` because each of these orderings needs atomic operations on
it. Two runtimes racing `set` on one slot need no atomic exchange, but only because the store and
the pre-read are seq_cst: a value overwritten without being pre-read was stored by a runtime that
still protects it (P4-dominance, as in channel 10) or pushed it when it stopped (P6, P12), since
the single total order rules out a pre-read older than a value whose writer stopped protecting it
after a test that saw marking off (P7).

`initField` has no barrier and a relaxed store, and is only for a cell no other thread can reach
yet: on a published slot it would also release none of this runtime's earlier records and stay out
of the single total order. A new cell's constructor must write every slot without a barrier, by
`initField` or by clearing it to `SharedValue::empty()` (`SharedArray`'s constructor uses a
`memset`): allocation returns uninitialised storage, so `set`'s pre-read would push whatever pointer
the previous occupant left, and the collector would follow it. `makeBoxed` publishes a cell's key
before it returns, and any cycle that starts after that can trace the cell, so every later write to
its slots is a `set`. `Atomics.exchange` and `compareExchange` will run the same pre-read barrier
before the operation, itself a seq_cst read-modify-write (channel 10; not yet implemented).

**P6 — Push before the prune erases** (implemented). The prune runs before a lookup
(`findBoxForSharedCell`) and after a box allocation (`makeBoxed`, the materialiser), and erases keys
whose box the local GC has found dead. With marking on it must push each such key before taking the
map lock. A dead box does not make the key safe to drop: while the box was alive, the runtime may
have stored the cell into a cell born this cycle, and a cell born this cycle is never traced. For
example, runtime R holds a box for Y, a cell older than the cycle. R allocates N (born marked) and
stores Y into N; the store barrier records N's old slot value, not Y. R drops Y's box, a local GC
clears the map entry, and R's next lookup or box allocation prunes Y's key before the collector
copies R's keys. Without the push nothing marks Y, Y is swept, and N's slot dangles. Not pruning
while marking is on would close the same hole, at the cost of keeping dead keys, and their cells,
for one more cycle. `Runtime::pruneSharedObjectBoxes` opens a barrier window, reads `cycleSeq()`
once and, if marking is on, pushes every dead key stamped with that value; it closes the window and
only then takes the map lock and erases. With no `SharedMutator` it pushes nothing. The map lock,
not the window, orders the pushes before a root scan that misses the erased keys. A prune whose test
sees marking off erases with no record. That is safe: the runtime cannot yet have stored the cell
into a cell born this cycle, and the test precedes the turn-on in the single total order, as do the
runtime's earlier slot stores, so every barrier that sees marking on pre-reads those values or later
ones (channel 7, P7). P6 also softens a breach of the raw-pointer rule: a prune while marking is on
pushes the key, so a raw pointer kept past its box stays valid through that cycle; a later cycle can
free the cell all the same.

**P7 — Reading the roots** (implemented). `markRuntimeRoots` copies every registered runtime's keys
(`SharedGC::copyRootKeys`) and marks only after releasing every lock. For each runtime in a snapshot
of the registry, the copy takes `barrierMtx_`, checks that the runtime is still registered, tries
its map lock and drops `barrierMtx_`, then copies the keys (`Runtime::appendSharedRootKeysUnlocked`)
under the map lock alone; a busy map is retried after a yield. Keys whose box has died but is not
yet pruned are copied too, which can keep a cell too long, never too short.

* The insert (`setBoxForSharedCell`, whose `operator[]` may rehash) and the prune (erase) take the
  map lock. The local GC's weak-root walk in `Runtime::markWeakRoots` does not: it rewrites values,
  never keys, so it never races the copy and no local GC pause waits for the collector.
* Copying one map at a time is meant to be sound: a key added after its map was copied belongs to a
  cell born after the reset (marked) or to a materialiser that saw marking on and pushed it (P4); a
  push that lands after certification is dropped and not needed, since the cell is still protected
  by its slot or by the runtime that stored it there ("Pushes that outlive their cycle"). A
  materialiser that saw marking off was inside a window when marking turned on, so the in-flight
  check waited for it and its key was in the map before the copy. The argument relies on the
  materialiser reading the pointer it publishes inside its window and publishing before the window
  closes, which its decided shape does; nothing enforces it: `setBoxForSharedCell` does not check
  that a window is open (boxing.md, "The collector's read: one lock per runtime").
* The root scan holds `barrierMtx_` across no copy, so a spill never waits for one, nor for a busy
  map. The map lock keeps the map alive through the copy: `unregisterMutator` takes it after the
  unlink, so a detaching runtime waits for a copy begun while it was registered, and no later scan
  finds it. A runtime that registers after the snapshot does so with an empty map, so the previous
  point covers every key it adds.
* **The edge must go both ways** (implemented; see Open questions: Model checking). P5's chain
  starts at the first writer whose barrier saw marking on. It needs every pre-read by a barrier that
  sees marking on to return no value older than one that another runtime stored and then stopped
  protecting after a test that saw marking off, and release and acquire alone do not forbid such a
  stale read. The wait gives only part of this. A release store read by the collector orders every
  runtime's earlier writes before the collector, and the second pass's read-modify-write orders the
  collector, with what it has learned, before every barrier whose `cycleSeq_` load reads it or a
  later value, but not before one whose load came before it. The single total order of seq_cst
  operations rules the stale read out for every barrier. Every slot store is seq_cst, and so are
  the turn-on and the barrier's test and pre-read. A test that sees marking on comes after the
  turn-on in that order, so a store that precedes the turn-on also precedes the pre-read, which then
  returns the stored value or a later one. That is the form barrier.md's case analysis relies on. A
  writer's later seq_cst test of `cycleSeq_` (in a prune window, or in `unregisterMutator` when the
  runtime detaches after `beginCycle` has created `markState_`) puts its earlier slot stores before
  the turn-on in that order if it saw marking off. A detach between cycles holds `barrierMtx_`,
  which `beginCycle` takes before the turn-on. A pre-read can still miss a value another runtime
  stored during the cycle, but only one that runtime still protects or pushed when it stopped (P6,
  P12), so the miss is harmless.

**P7a — Rings, spill and steal** (implemented). A barrier appends to its runtime's 256-entry
`SharedBarrierRing`; each entry carries its cycle stamp. The only producer is the thread currently
running that runtime, and it pushes without a lock. A full ring spills into the barrier worklist
under `barrierMtx_`, and certification steals what is left; the lock keeps the two consumers, the
spilling runtime and the stealing collector, apart. A runtime never does collector work to make
room, as a Hades mutator does when the shared barrier worklist is full (`handleFullBarrierChunk`
then runs `incrementalMark(0)`): a runtime blocked on a collector that is waiting for it would
deadlock.

**P8 — Black allocation and the reset** (implemented). `finishAlloc` marks every new cell under
`mtx_`, whether or not marking is on. `beginCycle` clears every mark bit and turns marking on
together, under `mtx_`.

* A mark means "reachable" and also "already queued; do not trace its children". Without the reset,
  birth marks, or marks left by an abandoned cycle, would hide those cells' children from the trace.
* No cell may be born between the reset and the turn-on. It would keep its mark and not be traced
  this cycle, so if a runtime stored Y into it and dropped its own key for Y before marking turned
  on, nothing would record Y.
* The mark is unconditional because a cell born between certification and the next sweep is marked
  by nothing else.
* A cell born this cycle is not traced, so its outgoing edges rest on P4-dominance (defined in the
  summary).
* `mtx_` is separate from `barrierMtx_`: they are taken at different rates, and merging them would
  tie allocation to the collector's drain. One global allocation lock is fine while shared
  allocation is rare.

**P9 — Atomic mark bitmap** (implemented). Each segment header has one mark bit per 8-byte granule,
in `std::atomic<uint8_t>` words (one word covers 64 bytes of heap), so a cell pointer finds its bit
by mask and shift. The collector and black allocation set bits with a relaxed `fetch_or`. It is
atomic because the collector marks without `mtx_` while allocation may set a neighbouring bit in the
same word. `beginCycle` clears all bits; the sweep clears survivors'. No epoch is needed: `mtx_`
gives the reset a safe moment.

**P10 — Termination** (implemented). A round certifies only if:

* (i) the collector's local worklist is empty on entry (a precondition, asserted: the caller has
  just drained it);
* (ii) one observation under `barrierMtx_` of the barrier worklist and every ring finds nothing that
  was not already marked, and marking is turned off inside that same critical section;
* (iii) the trace, which comes before that observation, reads every slot with an acquire load, so
  the observation sees every push made before a value the trace read, along the chain of writers
  to that slot (P5). A writer whose test saw marking off makes no pre-read, so the chain stops at
  its store; the two passes of the wait (P1) make the pushes of the writers before it visible.

`certifyRound` first marks the spilled backlog outside the lock. Clause (ii) must be one
observation: checked as "worklist empty, then the rings", a runtime can spill an unmarked cell into
the worklist just checked, and the cell is swept. Because marking turns off inside it, no spill
lands between the last drain and the turn-off, and any later spill keeps nothing. The observation
must also follow the trace, as (iii) says: otherwise the rings can be drained while empty, two
runtimes then push C and clear `X.g`, the trace reads null, and C is swept with its pushes still in
the rings.

The measure is "nothing newly marked", not emptiness: `for(;;){ S.f = a; S.f = b; }` refills the
rings forever with cells already marked. Each failed round marks a new cell, so rounds are bounded
by the live set. The test runs on the collector: runtimes pay nothing and are never waited for.
Turning marking off needs no wait, since later pushes are stale and dropped. `collect` sweeps on the
thread that set the marks, so program order publishes them to the sweep, and `mtx_` publishes the
birth marks (allocator.md, "What publishes the marks to the sweeper").

**P11 — Materialiser contract** (implemented in `Runtime::materializeSharedSlot`, which only tests
call yet). Push per P4. Read a map hit with the barriered `WeakRoot::get`, as `findBoxForSharedCell`
does, never `getNoBarrierUnsafe`: `HadesGC::weakRefReadBarrier` enqueues the box for the
old-generation marker while old-generation marking runs, and skipping it can hand out a box that
marking has already passed over. On a miss, allocate the box, look the key up again, and assign in
place with `operator[]`, as `setBoxForSharedCell` does. `DenseMap::insert` leaves an existing key
alone, so ignoring its return value breaks dedup; erase-then-insert would briefly remove the key
from the root set. Return the box as a `PseudoHandle` or `CallResult`, never as a `Handle` into the
materialiser's own `Locals`. The materialiser checks for a live hit outside any window, using the
slot's value only as a lookup key. On a miss it allocates the box and prunes, then reads the pointer
it publishes inside one window, which it enters with a seq_cst store and in which it must load
`cycleSeq_` with seq_cst before reading the slot, push before taking the map lock (a push can spill
under `barrierMtx_`, which the root scan takes before a map lock), and publish before the window
closes. Its caller keeps the cell that holds the slot alive across the call, by holding that cell's
box, since the box allocation can run a local GC and a whole cycle can pass before the window opens;
and it calls from outside any window, holding neither `mtx_` nor a map lock.

**P12 — Detach** (implemented). The Runtime constructor registers a post-shutdown deleter that
resets `sharedMutator_`. Not a shutdown callback: those run first and may call into JS and refill
the map. Post-shutdown deleters run after `finalizeAll()`, and finalizers never touch shared slots
("State and locks"), so nothing between the shutdown callbacks and the deleter pushes onto the ring.
`unregisterMutator` drains the ring into the barrier worklist rather than discarding it, since an
entry may be the only record of a cell another runtime still reaches. While marking is on it also
appends the runtime's keys to the barrier worklist. It copies them (`appendSharedRootKeysUnlocked`)
before it takes `barrierMtx_`, so a spill waits at most for the append, a memcpy, not for a walk of
the map, and without the map lock, since nothing inserts or erases while a runtime detaches and the
root scan only reads; only under `barrierMtx_` is it known whether marking is on, so the copy is
made either way. Leaving takes every key out of the root set at once, which opens P6's hole for each
of them: the runtime may already have stored one of those cells into a cell born this cycle, which
is never traced. The drain, the append and the unlink are one critical section under `barrierMtx_`,
so the collector never steals from a runtime being destroyed, and it finds either the runtime, in
the root scan, or its keys, in the barrier worklist that certification drains. After the unlink,
`unregisterMutator` takes the map lock and releases it, which waits for a root scan still copying
the map. With marking off nothing is appended. A runtime that leaves before marking turns on cannot
have stored into a cell born in that cycle, and every barrier that sees marking on pre-reads its
earlier slot stores or later values: if `markState_` exists, `unregisterMutator`'s seq_cst test of
`cycleSeq_` puts those stores before the turn-on in the single total order, and if not,
`barrierMtx_`, which `beginCycle` takes before the turn-on, orders them (channel 7, P7). After
certification every cell a runtime's keys name is already marked, by the root scan, the trace, a
push or at birth, while `endCycle` requires an empty barrier worklist.

### Acquisition and loss channels

Every way a shared cell can become reachable or unreachable during a cycle. Channels 4 and 5 are
closed by discipline rather than by a barrier.

| # | Channel | Closed by | Status |
|---|---|---|---|
| 1 | Box materialisation | P4, P7, P1's wait | implemented; only tests call it |
| 2 | Shared allocation | P8, `makeBoxed` | implemented |
| 3 | Shared string or BigInt read | v1: same as 1 | not yet implemented |
| 4 | Raw `SharedGCCell *` in a C++ local | the raw-pointer rule | discipline |
| 5 | Cross-runtime handoff | the raw-pointer rule | discipline |
| 6 | Root deletion during the cycle | P6 (a cell born this cycle may still hold the pointer) | implemented |
| 7 | Root deletion before the cycle | the total order over seq_cst slot stores, `cycleSeq_` tests and pre-reads (P7); `barrierMtx_` for a detach between cycles (P12) | implemented |
| 8 | Heap-edge deletion | P5 | implemented |
| 9 | Heap-edge insertion | P4-dominance | implemented: the allocation, acquisition (P4) and map-key (P6, P12) sides |
| 10 | `Atomics` read-modify-write | P5's seq_cst pre-read, before a seq_cst operation | not yet implemented |
| 11 | `SharedArray` indexed store | P5, through `SharedValue::set` | barrier implemented; JS routing not |
| 12 | Runtime teardown mid-cycle | P0, P12 | implemented |

**1.** Without the wait, counterexample CE-1. A raw pointer held across `BoxedSharedObject::create`
would leave the same gap with the wait; the materialiser's decided shape closes it by allocating the
box first and reading the pointer it publishes inside a window, which is what the in-flight check
waits for (boxing.md, "Acquisition: the materialiser").

**2.** The cell is born marked and published before `mtx_` is released, so it needs no push. A raw
`makeA` cell that is never boxed loses its birth mark at the next reset; only tests do that.

**3.** In v1 strings take the Object slow path (boxing.md's shared-strings appendix), so this is
channel 1.

**4.** Two cheap reinforcements: have every slot read that hands a cell to VM code return a box, as
the materialiser does (a `PseudoHandle<BoxedSharedObject>`, P11), never a raw `SharedGCCell *`; and
state the rule in `doc/GCSafeCoding.md`, which does not mention the shared heap.

**5.** A raw pointer crossing runtimes:

```
H2:  queue.send(box2->sharedCell_);   // no barrier runs
H2:  O2.f = null;                     // H2 drops its box; a later prune erases the key
H1:  c = queue.recv(); materialise(c);
```

If H2's prune erases the key while marking is off, the cell is in no map and on no ring until H1
materialises it, and a whole cycle can run in between. The rule makes the sender keep its box alive
until the receiver has boxed the cell.

**6.** Clearing a JS reference to a box (`O.f = null`) needs no barrier: the root is the map key,
which lives until the prune, and while marking is on the prune pushes each key before erasing it
(P6). So an ordinary property store costs nothing extra.

**7.** A key erased by a prune whose test saw marking off belongs to a dead box, and any heap path
to the cell is traced normally, since every older cell starts the cycle unmarked. That needs one
more guarantee: no barrier that sees marking on may pre-read a value older than one this runtime
stored before that test, or it could overwrite the cell's last edge unrecorded. The single total
order over the seq_cst slot store, the prune's test, the turn-on, the barrier's test and its
pre-read gives it. A runtime that leaves before marking turns on gets the same from
`unregisterMutator`'s seq_cst test, or from `barrierMtx_` between cycles (P7, P12). Re-acquiring is
channel 1, with a push.

**9.** A runtime boxes everything it allocates (boxing.md), so a string or BigInt migrated into the
shared heap by a store is allocated and boxed like any other cell. The two routes, plus the channel
5 handoff, are complete.

**10.** Inside a window, run P5's pre-read barrier, with its seq_cst pre-read, before the operation,
and make the operation a seq_cst read-modify-write. The recorded value may differ from the returned
one:

```
Initially S.f = Y.
A: pre-read Y; push Y
B: pre-read Y; push Y; exchange(S.f, Z) -> Y
A: exchange(S.f, null) -> Z
```

Both record Y, and Z was protected before B stored it (B's map entry, P4 or P8). In general, the
first modification that removes a snapshot edge to Y pre-read Y and pushed it before modifying; a
runtime that removed the edge in between would have been first. Later operations may replace
values other than their pre-reads, but each such value was protected before it was inserted, and
its writer still protects it or pushed it when it stopped; P7's two-way edge rules out the rest.
This first-remover argument is why no record of the returned value and no CAS retry loop are
needed. Separately, a run of read-modify-writes continues each one's release sequence, so a
collector that reads a later value also synchronizes with every earlier operation in the run and
sees its push; plain stores rely on P5's chain. A failed compare-exchange changes no edge; its push
only keeps a cell one cycle longer.

**11.** `SharedArray::set` goes through `SharedValue::set`, and `SharedArray::initField` is the
unbarriered form. JS indexed access must route through the seven `ObjectVTable` entries to these
(boxing.md O-B7).

### Cost

* **JS that never touches a shared object pays nothing.** A shared slot is reachable only through
  `box->sharedCell_`, which no `CellKind`, hidden class or inline cache describes. This assumes two
  things not yet enforced: the type inferrer never gives a box a user class type (`PrStore` would
  write its slots raw), and `SharedArray` indexed access routes through the hook.
* **Marking off:** a shared store adds its window's two stores (see "The wait" below), one seq_cst
  load and a branch, and its slot store is seq_cst: an `xchg` on x86, making two per `set` with the
  window's entry; on AArch64 the same `stlr` a release store uses; on ARMv7 a `dmb` on each side of
  the store. Give `cycleSeq_` its own cache line; today it sits next to `barrierMtx_`.
* **Marking on:** one seq_cst pre-read and one ring append per barrier, with no fence on x86 or
  AArch64 (on ARMv7 the pre-read and the ring's release store add a `dmb` each), and no atomic
  read-modify-write on the barrier path: barriers never touch mark bits.
* **The wait:** at most one barrier site per runtime in each of its two passes, with the exception
  under "Idle runtimes" above. The check costs two stores per outermost site, marking on or off: a
  seq_cst one at entry (`xchg` on x86; `stlr` on AArch64, no `dmb`) and a release one at exit (a
  plain store on x86, `stlr` on AArch64). On ARMv7, which Hermes ships, the check adds three `dmb`s
  to the off path. barrier.md covers a later asymmetric-fence option. The collector waits by
  polling, with `std::this_thread::yield()` between polls, in both passes and when the root scan
  finds a map busy; there is no backoff or limit, and each poll takes `barrierMtx_`. A runtime
  descheduled inside a window, or while it holds its map lock, keeps the collector thread spinning
  until it runs again. Whether to back off or block is open.
* **What dominates:** the local GC's walk of the map, one `acceptWeak` per entry (roughly 1-3 ns)
  on every local GC. It is O(map size) and independent of young-gen size, so the pause controller
  cannot shrink it; it is a cost of boxing rather than of marking. Spill and steal hand work over
  rather than stall.

---

## Rationale

### Counterexample CE-2: a runtime must not write mark bits

```
t1    A materialises a box for Q. A test-and-set on Q's mark bit returns 0, so A
      must queue Q.
t1.5  A is preempted before queuing it.
t2    The collector drains every buffer it can see, finds nothing new, certifies.
t3    Sweep. Q survives, since its bit is set, but was never traced, so Q.g's
      referent is freed.
```

The cell looks live and the damage lands one level down, where only the `HERMES_SLOW_DEBUG` re-trace
in `endCycle` can see it (Open questions: A safety net for P10). Any queue has the same two-step
window. So runtimes only push. Only the collector tests a mark before setting it, so a plain load
followed by `fetch_or` is enough; black allocation only sets the bit of a cell it is creating.

### Why SATB, not an insertion barrier

An insertion (Dijkstra) barrier records the value written. It is sound only if runtime roots are
rescanned at the end under a global stop, or kept black after one scan by a barrier on every write
to a root. Hermes has neither: there is no global stop and no rescan
(`Runtime::markRootsForCompleteMarking` marks only the sampling profiler), and a raw
`SharedGCCell *` in a C++ local is invisible to any scan or barrier. V8's shared-heap insertion
barrier relies on a global safepoint that re-derives every root after the last mutation. SATB keeps
everything reachable at the snapshot (Yuasa). A late store-barrier push is covered by P5's chain,
and a late acquisition push is redundant, since the slot's snapshot or its writer's key protects the
cell ("Pushes that outlive their cycle"). P4-dominance makes an insertion barrier redundant on top.

### Why the slot store and pre-read are seq_cst

**The slot store is a release,** so it cannot become visible before the push: a reader that loads
the new value with acquire sees the push too. Without that the collector could trace the cell, read
the new value, steal a ring whose tail has not moved, and certify, losing the old value with only
one runtime involved. `GCPointer<T>::set` states the same rule for local pointers: "Write barrier
must happen before the write." The collector's side is P10 (iii), the tracer's acquire loads.

**The pre-read is an acquire,** which carries that pairing along the chain of writers to one slot.
A release store pairs only with an acquire load that reads its value, or a later value in its
release sequence, and a store by a second runtime ends that sequence. But a second writer's
pre-read that reads the first writer's value synchronizes with that store, and its own store
releases what it has learned, and so on along the chain. barrier.md's chain lemma and case analysis
show that a round cannot certify while an overwritten value is unmarked.

**Every other slot read that may be followed to a cell is an acquire too:** the copy constructor
(so `SharedArray::at`), `getPointer`, and `getPointerOrNull`, which the tracer uses. So
the same release publishes a newly built cell to every reader, the collector included (boxing.md
O-B10). `isEmpty` and `isPointer` stay relaxed: nothing is read through their answer.

**Both are seq_cst, not only release and acquire,** for P7's two-way edge. A runtime may store a
cell into a slot and then stop protecting it, by pruning its key or detaching, after a test that
saw marking off. A barrier in another runtime that sees marking on must then pre-read that value or
a later one: if it read an older one, it would record that, overwrite this one unrecorded, and the
cell could be freed while a runtime still reaches it. Release and acquire allow that stale read.
With the slot store, the writer's test, the turn-on, the barrier's test and its pre-read all
seq_cst, the single total order rules it out.

**No fences.** A seq_cst store is already a release and the slot reads are already acquires, so a
release fence before the store, an acquire fence after the pre-read and an acquire fence in
certification would add nothing. Acquire and release fences around relaxed slot accesses cannot
give the total order, and they run only while marking is on, so they publish no new cell while it
is off. Cost: on x86 the seq_cst slot store is an `xchg`, making two per `set` with the window's
entry; on AArch64 it is the same `stlr` a release store uses; and with marking on the barrier runs
no fence on either.

With those fences instead, around relaxed slot accesses, two plain passes of the in-flight check,
with no read-modify-write between them, appear correct on x86-TSO, ARMv8 and `dmb`-mapped ARMv7
under the standard compilation mappings. The argument at the C++ level needs both the seq_cst slot
accesses and the read-modify-write, so the code does not rely on the hardware.

---

## Rejected alternatives

* **Each runtime turning its barriers on at its own next local GC:** counterexample CE-1 stays
  possible until every runtime has had one. One global `cycleSeq_` instead.
* **Each runtime confirming, from its own local GC, that it has seen marking on** (P2, retired):
  `Runtime::markWeakRoots` would store the odd `cycleSeq_` value it read, and the collector would
  wait for every runtime. A runtime that is idle, blocked in native code, running JS that does not
  allocate, or in Hades's promotion mode before TTI (whose young-gen collections skip
  `markWeakRoots`) never confirms, and Hermes has no safepoint to force a local GC, so the collector
  could wait forever. It would also be wrong for a window that can run a local GC, which would
  confirm from inside the window.
* **A 0/1 in-flight flag:** an inner site's exit clears it while the outer site is still in flight,
  and a busy runtime may be inside a barrier whenever the collector looks, so waiting for 0 can
  starve.
* **Mutate-then-check** (store, then test marking and push if it is on): the later test can see
  marking already off again, or a later cycle. See barrier.md's rejected alternatives.
* **Holding `mtx_` across the materialiser:** it allocates a local box, which can run a local GC, so
  every runtime's shared allocation would stall during someone else's local GC.
* **A per-barrier counter checked at termination** (even and unchanged across the steal): it
  brackets the push, not the store; relaxed increments prove nothing; and it waits while holding
  `barrierMtx_`, which deadlocks with `spill()`. P5's chain covers termination instead. P1's counter
  is written seq_cst on entry, read seq_cst once per runtime in each pass, and spans the whole site,
  and the collector waits for any change.
* **Reusing Hades's barrier chunk** (`barrierPushChunk_`, `barrierWorklist_`): it drains into the
  local mark worklist, and only the mutator flushes the chunk: a full one as it fills, the last
  partial one only from `completeMarking`, with the world stopped. The shared collector reads rings
  from another thread.
* **A separate root log:** unnecessary, since the collector copies each map's keys under its lock;
  a one-way log also would not give P7's two-way edge.
* **Pausing the prune during a cycle:** the map would grow meanwhile, and erasing from a `DenseMap`
  never gives capacity back.
* **Fences around the push, with relaxed slot accesses** (an acquire fence after the pre-read, a
  release fence before the store, and an acquire fence before certification's ring reads): they
  carry pushes to the collector, but keep the slot accesses out of the total order P7 needs, and
  publish no new cell while marking is off. See "Why the slot store and pre-read are seq_cst".

---

## Open questions

Most important first.

* **Model checking.** The two-pass wait (P1), P7's two-way edge and the acquisition push's
  redundancy (P4) rest on an argument made on paper under the C++20 memory model, backed by bounded
  GenMC models under RC11 and IMM: one or two tracked cells, three or four runtimes acting once, at
  most two cycles. The models find no lost cell, and variants that drop a needed ordering, a pass or
  the read-modify-write lose one, so they can see a loss: among them, a store barrier whose slot
  store and pre-read are only release and acquire. They check the design, not the code.
* **The materialiser's shape is not enforced** (channel 1; P11): `setBoxForSharedCell` does not
  check that it is called inside the materialiser's window, and the raw routes the tests use bypass
  the materialiser altogether (boxing.md O-B12).
* **`Atomics` integration:** each operation's memory ordering beyond channel 10's (a seq_cst
  read-modify-write after P5's seq_cst pre-read), how an operation reaches a struct field, boxing
  the returned value, and the `ObjectVTable` routing for `SharedArray` (boxing.md O-B7).
* **Trigger tuning.** The collector thread's rules (allocator.md, "Budget, mappings and exhaustion")
  are guesses: start at twice the footprint the last sweep left, or at that footprint plus
  `minTriggerBytes` (4 MiB) if that is more, but no more than halfway from it to the budget; start
  once `maxGap` (30 s) has passed since the thread's last cycle ended, if anything was allocated or
  a runtime left since the last cycle began; and leave a gap that keeps the collector busy at most
  half the time. Measure them on a real workload.
* **A safety net for P10.** Sweep frees rather than whitens, so one missed barrier is silently fatal
  and certification still finds nothing new. Builds with `HERMES_SLOW_DEBUG` check every certified
  cycle: `endCycle` re-traces from the box-map keys and asserts that every cell it reaches is marked
  and still allocated, and `collect` asserts after its sweep that no key names a freed cell. The
  check is one-way, since floating garbage means the two mark sets never agree, and it sees only
  box-map keys and slots. Still open: a guard for release builds, such as never freeing a cell in
  the cycle in which it was last seen reachable.
* **Barrier worklist growth** (barrier.md, open question 7). Nothing bounds `barrierWorklist`
  while marking is on: every store pushes the value it overwrites, marked or not, and `collect`
  drains the worklist only in `certifyRound`, never during the in-flight waits or the trace.

---

## Future options

Recorded, not planned.

**Thread-local allocation buffers (TLABs) with chunk-granular black allocation**, if shared
allocation stops being rare or `mtx_` shows up in a profile. Only a buffer refill takes `mtx_`, and
black allocation is recorded per chunk at refill: a chunk filled during a cycle counts as live at
sweep, and a reused chunk is recorded like a fresh one (a per-segment watermark would not survive
free-list reuse). Runtimes then never touch the mark bitmap, so its bits need not be atomic (P9),
the reset needs no lock (P8), and allocation is lock-free. Cost: a whole chunk is retained for a
cycle if any cell in it was allocated during that cycle, bounded by buffer size times the number of
runtimes. Design it with allocator.md's "Per-runtime local allocators" option.

**Split mark bitmaps**, if the collector's atomic `fetch_or` shows up in marking throughput and
TLABs are not an option: a mark bitmap written only by the collector, and an allocated-this-cycle
bitmap written only under `mtx_`. Sweep frees a cell only if it is in neither. No atomics on either
path, at the cost of a second bitmap and a second test in the sweeper.
