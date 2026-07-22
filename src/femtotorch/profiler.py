"""
femtotorch profiler (mostly AI generated).

Finds the bottleneck that makes a training step slow. There are only four
possible reasons:

  compute bound     the CPU is doing math at full speed, nothing is wasted
  bandwidth bound   the math is fast but the RAM delivers data too slowly
  dispatch bound    the time goes to python bookkeeping (Tensor, Node, calls)
  capacity bound    the RAM is full, the OS swaps to the SSD, everything stalls

Method: first measure the MAXIMUM this machine can do — one big matmul gives
the top math speed (GFLOP/s), one big array add gives the top data speed
(GB/s). Then compare what every operation actually achieved against those
maxima: near 100% of math speed -> compute bound; near 100% of data speed ->
bandwidth bound; page faults during the step -> capacity bound; far below
every maximum -> the time went to python (dispatch / under-used).

Usage:
    from femtotorch.profiler import Profiler
    Profiler().profile_model(model, X, y)    # X: input batch, y: class indices

    python -m femtotorch.profiler                    # demo MLP
    python -m femtotorch.profiler calibrate --force  # re-measure the maxima

The printed report has five sections; the Profiler class below has the same
blocks in the same order, plus [6] which runs everything:

  [1] machine maxima   the reference points (cached in ~/.femtotorch/)
  [2] latency          stopwatch on forward / backward
  [3] per-op trace     every op classified + verdict: where the step time goes
  [4] python hotspots  which function/line the time concentrates in
  [5] memory           does it fit in RAM, who allocates the most

Phases run SEPARATELY because each instrument distorts the others
(tracemalloc ~2x slowdown, cProfile ~1us per call, tracer ~1us per op).

Known limits, accepted rather than hidden: exp/log are counted as 1 flop per
element (understated); the graph-bytes counter counts references, not unique
buffers; trace() needs the v2 engine; BLAS threads are pinned to 1
(FEMTO_PROFILE_THREADS to override) — the CLI re-execs itself for that,
training scripts must set the env vars before python starts.
"""

# ------------------------------------------------------------------ thread pinning
# BLAS reads its thread-count env vars when numpy loads, and femtotorch's
# __init__ loads numpy before this module runs — so the pin only works if the
# vars were already set when python started. We record whether that happened:
# the CLI re-execs itself to fix it, Profiler() warns scripts that run unpinned.
import os
import sys
import warnings

_PIN = os.environ.get("FEMTO_PROFILE_THREADS", "1")
_PIN_VARS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS")
# this module runs twice under `python -m` (as femtotorch.profiler, then as
# __main__); only the first run can judge the pin, its verdict is stashed here
if "_FEMTO_PIN_OK" not in os.environ:
    _ok = (all(os.environ.get(v) for v in _PIN_VARS)
           or ("numpy" not in sys.modules and "cupy" not in sys.modules))
    os.environ["_FEMTO_PIN_OK"] = "1" if _ok else "0"
_PIN_OK = os.environ["_FEMTO_PIN_OK"] == "1"
for _var in _PIN_VARS:
    os.environ.setdefault(_var, _PIN)

import cProfile
import gc
import json
import platform
import pstats
import resource
import time
import tracemalloc
from pathlib import Path

from femtotorch.backend import xp, GPU, synchronize
from femtotorch import engine
from femtotorch import operations as ops
from femtotorch.engine_switch import Tensor, no_grad


# ------------------------------------------------------------------ FLOP counting
# Exact math count per op, from the array shapes. Movement ops (reshape, pad,
# im2col...) are 0 on purpose: they only move bytes, so their GB/s judges them.

def _matmul_fwd_flops(arrays, out, params):
    a, _b = arrays
    # (..., m, k) @ (..., k, n): each output scalar costs k mults + k adds
    return 2.0 * out.size * a.shape[-1]

_FWD_FLOPS = {
    "Add":    lambda arrays, out, params: float(out.size),
    "Mul":    lambda arrays, out, params: float(out.size),
    "Pow":    lambda arrays, out, params: float(out.size),
    "Relu":   lambda arrays, out, params: float(out.size),
    "Exp":    lambda arrays, out, params: float(out.size),
    "Log":    lambda arrays, out, params: float(out.size),
    "Sum":    lambda arrays, out, params: float(arrays[0].size),
    "Max":    lambda arrays, out, params: float(arrays[0].size),
    "Mean":   lambda arrays, out, params: float(arrays[0].size + out.size),
    "Matmul": _matmul_fwd_flops,
    # max + shift + exp + sum + div + log + gather, ~5 passes over the logits
    "SoftMaxCrossEntropy": lambda arrays, out, params: 5.0 * arrays[0].size,
    "Getitem": lambda arrays, out, params: 0.0,
    "Stack":   lambda arrays, out, params: 0.0,
    "PadZeros": lambda arrays, out, params: 0.0,
    "Reshape": lambda arrays, out, params: 0.0,
    "Swapaxes": lambda arrays, out, params: 0.0,
    "Im2col":  lambda arrays, out, params: 0.0,
}

def _matmul_bwd_flops(node, g, grads):
    # gA = g @ B^T and gB = A^T @ g: two gemms, exactly counted (this is where
    # the "backward = 2x forward" rule of thumb comes from, but we don't assume it)
    a, b = node.saved
    return 2.0 * a.size * g.shape[-1] + 2.0 * b.size * a.shape[-2]

_BWD_FLOPS = {
    "Matmul": _matmul_bwd_flops,
    # default for everything else: ~1 flop per produced gradient element
}


def _nbytes(values):
    """Sum nbytes of the array-like entries of an iterable, ignoring shapes/ints/None."""
    return sum(v.nbytes for v in values if hasattr(v, "nbytes"))


# ------------------------------------------------------------------ per-op tracer
class OpTracer:
    """
    The spy. Inside a `with` block, replaces every op's forward/backward
    (the ops.Function subclasses) with a version that stopwatches the real
    one and records time, FLOPs and bytes per op. No engine file is touched:
    the engine looks these methods up at call time, so swapping the class
    attributes from here is enough.

    Also tracks how many bytes the autograd graph is holding (everything
    stashed by Node.save): save() increments, Node.__del__ decrements,
    and the high-water mark is the exact peak the graph retained.
    """

    def __init__(self):
        self.stats = {}          # (name, dir) -> [calls, ns, flops, bytes]
        self.graph_bytes = 0     # currently retained by Node.save
        self.peak_graph_bytes = 0
        self._originals = []     # [(cls, attr_name, original_staticmethod)]

    def _record(self, name, direction, ns, flops, nbytes):
        entry = self.stats.setdefault((name, direction), [0, 0, 0.0, 0])
        entry[0] += 1
        entry[1] += ns
        entry[2] += flops
        entry[3] += nbytes

    # -------------------------------------------------- patching
    def _op_classes(self):
        stack, seen = [ops.Function], set()
        while stack:
            cls = stack.pop()
            for sub in cls.__subclasses__():
                if sub not in seen:
                    seen.add(sub)
                    stack.append(sub)
        return seen

    def _wrap_forward(self, cls):
        orig, tracer, name = cls.forward, self, cls.__name__
        flops_fn = _FWD_FLOPS.get(name)

        def forward(node, *arrays, **params):
            t0 = time.perf_counter_ns()
            out = orig(node, *arrays, **params)
            synchronize()
            ns = time.perf_counter_ns() - t0
            flops = flops_fn(arrays, out, params) if flops_fn else 0.0
            moved = _nbytes(arrays) + _nbytes(params.values()) + out.nbytes
            tracer._record(name, "fwd", ns, flops, moved)
            return out
        return staticmethod(forward)

    def _wrap_backward(self, cls):
        orig, tracer, name = cls.backward, self, cls.__name__
        flops_fn = _BWD_FLOPS.get(name)

        def backward(node, g):
            t0 = time.perf_counter_ns()
            grads = orig(node, g)
            synchronize()
            ns = time.perf_counter_ns() - t0
            if flops_fn is not None:
                flops = flops_fn(node, g, grads)
            else:
                flops = float(sum(gr.size for gr in grads if hasattr(gr, "size")))
            moved = g.nbytes + _nbytes(node.saved) + _nbytes(grads)
            tracer._record(name, "bwd", ns, flops, moved)
            return grads  # node's bytes are decremented by Node.__del__ when the engine frees it
        return staticmethod(backward)

    def __enter__(self):
        for cls in self._op_classes():
            for attr in ("forward", "backward"):
                if attr in cls.__dict__:  # skip inherited Function stubs
                    self._originals.append((cls, attr, cls.__dict__[attr]))
            if "forward" in cls.__dict__:
                cls.forward = self._wrap_forward(cls)
            if "backward" in cls.__dict__:
                cls.backward = self._wrap_backward(cls)

        tracer = self
        orig_save = engine.Node.save
        self._orig_save = orig_save

        def save(node, *values):
            orig_save(node, *values)
            nb = _nbytes(values)
            node._prof_saved_bytes = nb
            tracer.graph_bytes += nb
            if tracer.graph_bytes > tracer.peak_graph_bytes:
                tracer.peak_graph_bytes = tracer.graph_bytes

        def _node_del(node):
            tracer.graph_bytes -= getattr(node, "_prof_saved_bytes", 0)

        engine.Node.save = save
        engine.Node.__del__ = _node_del
        return self

    def __exit__(self, *exc):
        for cls, attr, original in self._originals:
            setattr(cls, attr, original)
        self._originals.clear()
        engine.Node.save = self._orig_save
        try:
            del engine.Node.__del__
        except AttributeError:
            pass
        return False


# ------------------------------------------------------------------ small helpers
def _time_best_s(fn, warmup=2, repeats=5):
    """Best-of-N wall time in seconds; best (not mean) because for a fixed
    workload the minimum is the least-perturbed sample."""
    for _ in range(warmup):
        fn()
        synchronize()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        synchronize()
        best = min(best, time.perf_counter_ns() - t0)
    return best / 1e9


def _percentile(sorted_values, q):
    idx = round(q / 100 * (len(sorted_values) - 1))
    return sorted_values[idx]


def _maxrss_bytes(ru_maxrss):
    # classic portability trap: macOS reports bytes, Linux reports kilobytes
    return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024


def _mb(nbytes):
    return nbytes / 1e6


# ------------------------------------------------------------------ the profiler
# plain-language names and advice for the final summary, one per bound
_BUCKET_LABEL = {
    "compute":    "compute    (math already at max speed)",
    "bandwidth":  "bandwidth  (limited by RAM speed)",
    "under-used": "under-used (ops too small for the machine)",
    "dispatch":   "dispatch   (python bookkeeping)",
}
_ADVICE = {
    "compute":    "machine already at max math speed -> do less math "
                  "(smaller model, fused ops); optimizing python won't help",
    "bandwidth":  "limited by RAM speed -> move fewer bytes "
                  "(fewer intermediate arrays, in-place ops)",
    "under-used": "ops too small to use the machine -> bigger batch: "
                  "same python cost, much more math per op",
    "dispatch":   "python bookkeeping dominates -> fewer but bigger ops "
                  "(bigger batch, fused ops) or a leaner engine",
}


class Profiler:
    """Six blocks, matching the six sections of the printed report."""

    _ROOF_CACHE = Path.home() / ".femtotorch" / "roofline.json"

    def __init__(self):
        self.roofs = None       # {'gflops_roof', 'gbps_roof', 'ridge_flops_per_byte'}
        self._dispatch = None   # {'framework_ns', 'kernel_ns', 'overhead_ns'}
        if not _PIN_OK:
            warnings.warn(
                "BLAS threads are not pinned (numpy was imported before the pin "
                "could apply): measurements will wobble with thread scheduling. "
                "Set the vars before launching python, e.g. "
                "VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python train.py")

    # ============================================================== [1]
    # MACHINE MAXIMA — the reference points every verdict compares against.
    #   calibrate():         top math speed (GFLOP/s) + top data speed (GB/s)
    #   dispatch_overhead(): the fixed python cost every femtotorch op pays
    # ==============================================================
    def calibrate(self, force=False, verbose=True, gemm_n=2048, stream_elems=32 * 2**20):
        """
        Measures the machine's two maxima THROUGH numpy (honest for this
        stack, unlike spec sheets): math speed via a big float32 matmul,
        data speed via a big streaming add. Their ratio is the "ridge":
        an op doing fewer flops per byte than the ridge can never be
        compute bound. Cached on disk — the machine doesn't change.
        """
        backend = "cupy" if GPU else "numpy"
        if not force and self._ROOF_CACHE.exists():
            cached = json.loads(self._ROOF_CACHE.read_text())
            if cached.get("threads") == _PIN and cached.get("backend") == backend:
                self.roofs = cached
                if verbose:
                    print(f"    math max {cached['gflops_roof']:.1f} GFLOP/s | "
                          f"data max {cached['gbps_roof']:.1f} GB/s | "
                          f"ridge {cached['ridge_flops_per_byte']:.1f} flops/byte (cached)")
                return self.roofs

        a = xp.random.rand(gemm_n, gemm_n).astype(xp.float32)
        b = xp.random.rand(gemm_n, gemm_n).astype(xp.float32)
        t = _time_best_s(lambda: a @ b)
        gflops_roof = (2.0 * gemm_n**3 / t) / 1e9
        del a, b

        a = xp.zeros(stream_elems, dtype=xp.float32)
        b = xp.ones(stream_elems, dtype=xp.float32)
        c = xp.empty(stream_elems, dtype=xp.float32)
        t = _time_best_s(lambda: xp.add(a, b, out=c))
        gbps_roof = (3.0 * a.nbytes / t) / 1e9  # read a, read b, write c
        del a, b, c

        self.roofs = {
            "gflops_roof": gflops_roof,
            "gbps_roof": gbps_roof,
            "ridge_flops_per_byte": gflops_roof / gbps_roof,
            "threads": _PIN,
            "backend": backend,
            "machine": platform.platform(),
        }
        self._ROOF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        self._ROOF_CACHE.write_text(json.dumps(self.roofs, indent=2))
        if verbose:
            print(f"    math max {gflops_roof:.1f} GFLOP/s | "
                  f"data max {gbps_roof:.1f} GB/s | "
                  f"ridge {self.roofs['ridge_flops_per_byte']:.1f} flops/byte "
                  f"(measured, {_PIN} BLAS thread(s))")
        return self.roofs

    def dispatch_overhead(self, calls=500, repeats=5, verbose=True):
        """
        The python tax: a femtotorch add on (1,1) tensors (wrapper + Node +
        kernel) minus a raw xp.add (kernel only). Every op costs at least
        this, no matter how small its arrays are.
        """
        ta = Tensor(xp.ones((1, 1), dtype=xp.float32))
        tb = Tensor(xp.ones((1, 1), dtype=xp.float32))
        xa = xp.ones((1, 1), dtype=xp.float32)
        xb = xp.ones((1, 1), dtype=xp.float32)

        def per_call_ns(f):
            best = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                for _ in range(calls):
                    f()
                best = min(best, (time.perf_counter_ns() - t0) / calls)
            return best

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            framework_ns = per_call_ns(lambda: ta + tb)
            kernel_ns = per_call_ns(lambda: xp.add(xa, xb))
        finally:
            if gc_was_enabled:
                gc.enable()

        self._dispatch = {
            "framework_ns": framework_ns,
            "kernel_ns": kernel_ns,
            "overhead_ns": framework_ns - kernel_ns,
        }
        if verbose:
            print(f"    python tax per op: {framework_ns / 1e3:.2f} us "
                  f"= {kernel_ns / 1e3:.2f} numpy kernel "
                  f"+ {self._dispatch['overhead_ns'] / 1e3:.2f} femtotorch bookkeeping")
        return self._dispatch

    # ============================================================== [2]
    # LATENCY — plain stopwatch on forward / backward, no instrumentation.
    # ==============================================================
    def latency(self, fn, setup=None, warmup=10, iters=100, label=None):
        """
        Stopwatch on fn, gc off, repeated `iters` times; median + p10/p90
        (median, because the mean is skewed by OS scheduling spikes).
        If setup is given, each round runs arg = setup() UNTIMED then times
        fn(arg) — how backward is timed without paying for forward:
            latency(lambda loss: loss.backward(), setup=make_loss)
        """
        times_ns = []
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(warmup):
                fn(setup()) if setup else fn()
            synchronize()
            for _ in range(iters):
                arg = setup() if setup else None
                t0 = time.perf_counter_ns()
                fn(arg) if setup else fn()
                synchronize()
                times_ns.append(time.perf_counter_ns() - t0)
        finally:
            if gc_was_enabled:
                gc.enable()

        times_ns.sort()
        result = {
            "median_ms": _percentile(times_ns, 50) / 1e6,
            "p10_ms": _percentile(times_ns, 10) / 1e6,
            "p90_ms": _percentile(times_ns, 90) / 1e6,
            "iters": iters,
        }
        if label:
            print(f"    {label:<22} median {result['median_ms']:8.3f} ms   "
                  f"(p10 {result['p10_ms']:.3f} / p90 {result['p90_ms']:.3f})")
        return result

    # ============================================================== [3]
    # PER-OP TRACE — the verdict. Spies on every op of ONE training step,
    # compares each op against the maxima of [1] -> compute / bandwidth /
    # dispatch / under-used, then sums up where the step time goes.
    # ==============================================================
    def _classify(self, per_call_ns, gflops, gbps):
        floor_ns = self._dispatch["framework_ns"]
        if per_call_ns < 3 * floor_ns:
            return "dispatch"
        frac_compute = gflops / self.roofs["gflops_roof"]
        frac_bandwidth = gbps / self.roofs["gbps_roof"]
        if frac_compute >= 0.5:
            return "compute"
        if frac_bandwidth >= 0.5:
            return "bandwidth"
        return f"under-used ({100 * max(frac_compute, frac_bandwidth):.0f}% of max)"

    def trace(self, step_fn, verbose=True):
        """
        Runs step_fn once with the OpTracer spy on, classifies every op
        against the maxima of [1], then sums up where the step time went.
        Wall time minus time inside the ops = the engine's own python cost
        (graph build, grad accumulation) — printed as "(engine overhead)".
        v2 engine only: v1's closures don't go through ops.Function.
        """
        if not Tensor.__module__.endswith("TensorV2"):
            raise RuntimeError("trace() hooks operations.Function, which only the "
                               "v2 engine uses — run with FEMTO_ENGINE=v2")
        if self.roofs is None:
            self.calibrate(verbose=verbose)
        if self._dispatch is None:
            self.dispatch_overhead(verbose=verbose)

        step_fn()  # warmup: allocator + caches settle, untraced
        with OpTracer() as tracer:
            t0 = time.perf_counter_ns()
            step_fn()
            synchronize()
            wall_ns = time.perf_counter_ns() - t0

        rows = []
        for (name, direction), (calls, ns, flops, nbytes) in tracer.stats.items():
            seconds = ns / 1e9
            gflops = (flops / seconds) / 1e9 if seconds > 0 else 0.0
            gbps = (nbytes / seconds) / 1e9 if seconds > 0 else 0.0
            rows.append({
                "op": name, "dir": direction, "calls": calls,
                "total_ms": ns / 1e6, "pct_of_step": 100 * ns / wall_ns,
                "gflops": gflops, "gbps": gbps,
                "bound": self._classify(ns / calls, gflops, gbps),
            })
        rows.sort(key=lambda r: r["total_ms"], reverse=True)
        traced_ns = sum(v[1] for v in tracer.stats.values())

        # summary: step time by cause; "under-used (x% of max)" -> "under-used"
        buckets = {"compute": 0.0, "bandwidth": 0.0, "under-used": 0.0,
                   "dispatch": (wall_ns - traced_ns) / 1e6}
        for r in rows:
            buckets[r["bound"].split(" ")[0]] += r["total_ms"]

        report = {
            "rows": rows,
            "wall_ms": wall_ns / 1e6,
            "traced_ms": traced_ns / 1e6,
            "framework_ms": (wall_ns - traced_ns) / 1e6,
            "framework_pct": 100 * (wall_ns - traced_ns) / wall_ns,
            "peak_graph_mb": _mb(tracer.peak_graph_bytes),
            "buckets": buckets,
            "advice": _ADVICE[max(buckets, key=buckets.get)],
        }
        if verbose:
            self._print_trace_report(report)
        return report

    def _print_trace_report(self, report):
        """Presentation only — every number comes from the report dict."""
        print(f"    one training step = {report['wall_ms']:.2f} ms, op by op:")
        print(f"    {'op':<20}{'dir':<5}{'calls':>6}{'total ms':>10}{'% step':>8}"
              f"{'GFLOP/s':>9}{'GB/s':>7}   bound")
        for r in report["rows"]:
            gf = f"{r['gflops']:.1f}" if r["gflops"] > 0 else "-"
            print(f"    {r['op']:<20}{r['dir']:<5}{r['calls']:>6}{r['total_ms']:>10.3f}"
                  f"{r['pct_of_step']:>7.1f}%{gf:>9}{r['gbps']:>7.1f}   {r['bound']}")
        print(f"    {'(engine overhead)':<20}{'':<5}{'':>6}{report['framework_ms']:>10.3f}"
              f"{report['framework_pct']:>7.1f}%{'-':>9}{'':>7}   dispatch")
        print(f"    (maxima of [1] for comparison: {self.roofs['gflops_roof']:.0f} GFLOP/s "
              f"math, {self.roofs['gbps_roof']:.0f} GB/s data)")
        print(f"    peak memory retained by the autograd graph: "
              f"{report['peak_graph_mb']:.2f} MB")
        print()
        print("    verdict — where the time of one step goes:")
        for key in ("compute", "bandwidth", "under-used", "dispatch"):
            pct = 100 * report["buckets"][key] / report["wall_ms"]
            bar = "#" * round(pct / 2.5)
            print(f"      {_BUCKET_LABEL[key]:<44}{pct:>5.1f}%  {bar}")
        print(f"    -> biggest lever: {report['advice']}")

    # ============================================================== [4]
    # PYTHON HOTSPOTS — cProfile: which exact function eats the time.
    # ==============================================================
    def cprofile(self, step_fn, top=12, repeat=20):
        """
        Hotspot map: which function/line the time concentrates in, sorted by
        cumulative time. Runs the step `repeat` times: one sub-ms step is
        below cProfile's display resolution.
        Caveats: cProfile adds ~1us per python CALL, and math invoked through
        operators (a @ b, a + b) is charged to the python frame that ran the
        operator, not to numpy — so no honest python-vs-C ratio exists here;
        the engine-overhead line of trace() is the trustworthy python share.
        Use this phase only to locate WHERE time goes, not to quantify it.
        """
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(repeat):
            step_fn()
        profile.disable()

        stats = pstats.Stats(profile)
        print(f"    over {repeat} steps, top functions by total time "
              f"('own' = inside the function itself, excluding what it calls):")
        print(f"    {'calls':>6}{'own ms':>9}{'with sub':>10}   function")
        entries = sorted(stats.stats.items(), key=lambda kv: kv[1][3], reverse=True)
        for (filename, lineno, func), (_cc, nc, tt, ct, _callers) in entries[:top]:
            where = func.strip("<>") if filename == "~" \
                else f"{Path(filename).name}:{lineno} {func}"
            print(f"    {nc:>6}{tt * 1e3:>9.1f}{ct * 1e3:>10.1f}   {where}")
        return {"entries": entries[:top]}

    # ============================================================== [5]
    # MEMORY — capacity: does the step fit in RAM, and who allocates most.
    # ==============================================================
    def memory(self, step_fn, top=8):
        """
        Three questions, three tools:
          - is RAM full?    rusage page faults (major faults = swapping)
          - how much does one step allocate?    tracemalloc peak
          - who allocates?  tracemalloc top file:line (numpy arrays are seen)
        Runs alone because tracemalloc ~doubles runtime. The RSS high-water
        covers the whole process life, calibration arrays included.
        """
        step_fn()  # warmup, untracked
        ru0 = resource.getrusage(resource.RUSAGE_SELF)

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        step_fn()
        _, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        result = {
            "step_alloc_peak_mb": _mb(peak - baseline),
            "rss_peak_mb": _mb(_maxrss_bytes(ru1.ru_maxrss)),
            "major_page_faults": ru1.ru_majflt - ru0.ru_majflt,
            "capacity_bound": ru1.ru_majflt - ru0.ru_majflt > 0,
        }
        print(f"    one step allocates {result['step_alloc_peak_mb']:.2f} MB at its peak "
              f"(process high-water {result['rss_peak_mb']:.0f} MB, incl. calibration)")
        if result["capacity_bound"]:
            print(f"    {result['major_page_faults']} major page faults -> RAM is FULL, "
                  f"the OS swaps to disk: memory capacity bound")
        else:
            print("    0 major page faults -> fits in RAM (not capacity bound)")
        print("    biggest allocators (file:line):")
        for stat in snapshot.statistics("lineno")[:top]:
            frame = stat.traceback[0]
            print(f"    {stat.size / 1024:>8.0f} KiB in {stat.count:>3} allocations   "
                  f"{Path(frame.filename).name}:{frame.lineno}")
        return result

    # ============================================================== [6]
    # FULL REPORT — every phase, in an order where instruments never overlap.
    # ==============================================================
    def profile_model(self, model, X, target, loss_fn=None,
                      warmup=10, iters=50):
        """
        The full report: sections [1] to [5], in an order where instruments
        never overlap. loss_fn(pred, target) -> scalar Tensor; defaults to
        fused softmax cross entropy + mean.
        """
        X = X if isinstance(X, Tensor) else Tensor(X)
        if loss_fn is None:
            loss_fn = lambda pred, t: pred.softmax_cross_entropy(t).mean()

        def fwd_infer():
            with no_grad():
                model.forward(X)

        def fwd_graph():
            model.forward(X)

        def make_loss():
            model.zero_grad()
            return loss_fn(model.forward(X), target)

        def step():
            loss = make_loss()
            loss.backward()

        width = 72
        print("=" * width)
        print("femtotorch profiler report".center(width))
        print("=" * width)

        print("\n[1] MACHINE MAXIMA — the reference points everything is compared to")
        self.calibrate()
        self.dispatch_overhead()

        print(f"\n[2] LATENCY — stopwatch, median of {iters} runs")
        results = {
            "fwd_no_grad": self.latency(fwd_infer, warmup=warmup, iters=iters,
                                        label="forward (no_grad)"),
            "fwd_graph": self.latency(fwd_graph, warmup=warmup, iters=iters,
                                      label="forward (+graph)"),
            "backward": self.latency(lambda loss: loss.backward(), setup=make_loss,
                                     warmup=warmup, iters=iters, label="backward"),
        }
        # graph-build tax: same math, only the autograd bookkeeping differs
        graph_tax = results["fwd_graph"]["median_ms"] - results["fwd_no_grad"]["median_ms"]
        bwd_ratio = results["backward"]["median_ms"] / results["fwd_graph"]["median_ms"]
        print(f"    -> graph build costs {graph_tax:.3f} ms "
              f"({100 * graph_tax / results['fwd_graph']['median_ms']:.0f}% of forward); "
              f"backward = {bwd_ratio:.1f}x forward")

        print("\n[3] PER-OP TRACE — what is each op limited by?")
        results["trace"] = self.trace(step)

        print("\n[4] PYTHON HOTSPOTS — which functions the python time is spent in")
        results["cprofile"] = self.cprofile(step)

        print("\n[5] MEMORY — does the step fit in RAM, and who allocates")
        results["memory"] = self.memory(step)

        print("\n" + "=" * width)
        return results


# ------------------------------------------------------------------ demo / CLI
def _demo(batch=256, iters=50):
    """MNIST-shaped synthetic MLP so the profiler can be exercised standalone."""
    from femtotorch.nn import MLP
    rng = xp.random.default_rng(0)
    X = rng.standard_normal((batch, 784)).astype(xp.float32)
    y = rng.integers(0, 10, size=batch)
    model = MLP(784, [256, 256, 256, 256, 256, 128, 10])
    Profiler().profile_model(model, X, y, iters=iters)


if __name__ == "__main__":
    # numpy loaded before the pin could apply -> restart python once with the
    # vars already in the environment. The marker prevents exec loops.
    if not _PIN_OK and os.environ.get("_FEMTO_PROF_REEXEC") != "1":
        os.environ["_FEMTO_PROF_REEXEC"] = "1"
        del os.environ["_FEMTO_PIN_OK"]  # child must judge the pin afresh
        os.execv(sys.executable,
                 [sys.executable, "-m", "femtotorch.profiler", *sys.argv[1:]])

    import argparse
    parser = argparse.ArgumentParser(description="femtotorch profiler")
    parser.add_argument("mode", nargs="?", default="demo", choices=["demo", "calibrate"])
    parser.add_argument("--force", action="store_true", help="recalibrate the roofline")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "calibrate":
        Profiler().calibrate(force=args.force)
    else:
        _demo(batch=args.batch, iters=args.iters)


# ============================================================================
# EPHEMERAL — two-engine comparison. DELETE THIS WHOLE BLOCK when done.
#
# The engine (closures = tensor.py / Node = TensorV2.py) is chosen at IMPORT
# time by FEMTO_ENGINE, so the only honest way to profile both is two separate
# processes. compare_engines() runs your CIFAR_VGG.py twice, once per possible
# FEMTO_ENGINE value, BLAS pinned. It does NOT trust the label: each child
# announces the engine it actually loaded (Tensor.__module__), so the run is
# correctly identified even while engine_switch.py's mapping is inverted.
# The child patch also makes section [3] SKIP on the old engine (whose closures
# the tracer can't hook) instead of aborting the whole report.
#
# Run it from the repo root (so CIFAR_VGG.py finds data/cifar10):
#     python -c "import femtotorch.profiler as p; p.compare_engines()"
#
# CIFAR_VGG.py runs its full body (one train batch -> profiler report -> a 10k
# test eval). The two reports you want print BEFORE each eval; let it finish, or
# read them as they appear.
# ============================================================================

# runs in each spawned child only
if os.environ.get("_FEMTO_COMPARE_CHILD") == "1":
    _is_new = Tensor.__module__.endswith("TensorV2")
    print(f"\n>>> real engine loaded: {Tensor.__module__} "
          f"({'NEW Node engine' if _is_new else 'OLD closure engine'})", flush=True)

    # the tracer can only hook the new engine; skip [3] on the old one
    _orig_trace = Profiler.trace

    def _trace_or_skip(self, step_fn, verbose=True):
        try:
            return _orig_trace(self, step_fn, verbose=verbose)
        except RuntimeError as exc:
            print(f"    skipped — {exc}")
            return None

    Profiler.trace = _trace_or_skip


def compare_engines(script="milestones_examples/CIFAR_VGG.py"):
    import subprocess

    pin = {v: "1" for v in _PIN_VARS}  # pin the children so numbers don't wobble
    for value in ("v1", "v2"):         # the two values cover both engines, whichever way the switch maps them
        print("\n" + "#" * 78)
        print(f"#  spawning {script}  (FEMTO_ENGINE={value})".ljust(77) + "#")
        print("#" * 78, flush=True)

        env = {**os.environ, **pin,
               "FEMTO_ENGINE": value, "_FEMTO_COMPARE_CHILD": "1"}
        env.pop("_FEMTO_PIN_OK", None)      # let the child judge the pin afresh
        env.pop("_FEMTO_PROF_REEXEC", None)
        subprocess.run([sys.executable, script], env=env)
