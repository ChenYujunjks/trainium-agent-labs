#!/usr/bin/env python3
"""
agent.py — the loop: a model writes an NKI kernel, the harness grades it, the reason goes back,
it tries again.

    controller  picks a level                          nkibench.py
    generator   the model writes a kernel             the shared gpt-oss endpoint
    checker     rules, then simulate, then roofline   nkibench.py
    loop        the failure becomes the next prompt   this file

Everything here runs on the CPU, so it does NOT need the Trainium device -- which is the point,
because the device is what Project 2 needs later for real timings, and a Neuron device cannot be
shared by two processes. The model therefore comes from the shared endpoint, not from a server on
your own chip.

    export GPTOSS_BASE_URL="https://..."
    python agent.py --level 1
    python agent.py --level 4 --rounds 6 --samples 4
    python agent.py --all
    python agent.py --offline --level 4        # no model; replays the reference kernel

Every attempt is appended to a JSONL file with its reward, so the log is the deliverable.
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time

import numpy as np

import nkibench

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")

# The model writes to a hidden reasoning channel before it writes any answer. Measured on this
# endpoint: a coding task burned 900 tokens thinking and returned EMPTY content. See gptoss/README.
MIN_ANSWER_TOKENS = 2500

REASONING_KEYS = ("reasoning", "reasoning_content")


# ---------------------------------------------------------------- reward
#
# Graded, not pass/fail, so a near miss is distinguishable from nonsense and the loop has
# something to climb. Matches Project 1's shape: correctness dominates, and nothing else counts
# until the kernel is right.

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)


def grade(source, level):
    """Returns (reward, parts, feedback). Feedback is an INSTRUCTION, never just a verdict."""
    parts = dict(parses=False, rules=False, runs=False, correct=False)

    if not source.strip():
        return 0.0, parts, ("No code came back. Reply with one python code block containing the "
                            "kernel and nothing else.")
    try:
        compile(source, "<candidate>", "exec")
        parts["parses"] = True
    except SyntaxError as e:
        return (WEIGHTS["parses"] * 0, parts,
                f"The code does not parse: {e.msg} on line {e.lineno}. Send one complete python "
                f"code block.")

    violations = nkibench.check_rules(source, level)
    if violations:
        extra = ""
        if any("no function named" in v for v in violations):
            # Measured: this repeated 15 rounds running, because "there is no function named X"
            # never said what the function should look like. Hand over the exact line.
            import inspect
            ref = nkibench.LEVELS[level]["ref"]
            args = ", ".join(inspect.signature(ref).parameters)
            extra = (f" Start the function with exactly this line:  "
                     f"def {nkibench.LEVELS[level]['entry']}({args}):  "
                     f"and put @nki.jit on the line above it.")
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                "Rule violations, which score zero however fast the kernel is. Fix exactly "
                "these: " + " ".join(violations) + extra)
    parts["rules"] = True

    spec = nkibench.LEVELS[level]
    path = f"/tmp/_agent_level{level}.py"
    with open(path, "w") as f:
        f.write(source)
    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except ModuleNotFoundError as e:
        # Two very different causes, and conflating them aborted a whole run: either the SDK is
        # absent (an environment problem the model cannot fix), or the model imported a module it
        # invented -- observed: `import nki.nl`. Tell them apart by asking whether nki itself is
        # importable.
        try:
            import nki  # noqa: F401
            sdk_present = True
        except ImportError:
            sdk_present = False
        if not sdk_present:
            raise SystemExit(
                f"cannot import {e.name!r}, so no NKI kernel can be loaded here. Run this where "
                f"the Neuron SDK exists -- k8s/kernel-agent-job.yaml does that -- or use "
                f"--offline to exercise the loop without a kernel ever running.") from e
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"There is no module named {e.name!r}. The only imports that exist are: "
                f"`import nki`, `import nki.language as nl`, and `import nki.isa as nisa`. "
                f"Use exactly those three.")
    except Exception as e:
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"The file imports but {spec['entry']} could not be loaded: "
                f"{type(e).__name__}: {e}")

    failures, passed, intensity = [], 0, None
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"CANNOT SIMULATE: {e}")
        except Exception as e:
            err = f"raised {type(e).__name__}: {e}"
            hint = shape_hint(err, args, level) if SHAPE_HINT else ""
            hint = hint or (reshape_hint(err, args, level) if RESHAPE_HINT else "")
            failures.append((nkibench.label(case, level), hint or enrich(err)))
            continue
        parts["runs"] = True
        m = (nkibench.check_inputs_untouched(before, args)
             or nkibench.describe_mismatch(got, want)
             or nkibench.check_traffic_bar(level, counted, args, want))
        # A simulator warning about a hardware-correctness hazard counts as a failure even when the
        # numbers happen to match on CPU: the kernel would be wrong on the device.
        hazards = [w for w in counted.get("warnings", [])
                   if "incorrect results on hardware" in w]
        if hazards and not m:
            m = ("CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0]
                 + ". Fix that before anything else -- the simulator agrees with the reference here "
                   "and the device would not.")
        if m:
            failures.append((nkibench.label(case, level), m))
            continue
        passed += 1
        if level >= 3 and counted["bytes"]:
            intensity = nkibench.roofline(
                nkibench.matmul_flops(case["M"], case["K"], case["N"]), counted["bytes"])

    if failures:
        lbl, first = failures[0]
        return (sum(WEIGHTS[k] for k, v in parts.items() if v)
                + WEIGHTS["correct"] * passed / len(spec["shapes"]), parts,
                f"{passed} of {len(spec['shapes'])} shapes passed. On {lbl}: {first}")

    parts["correct"] = True
    reward = sum(WEIGHTS.values())
    note = "Correct on every shape."
    if intensity:
        note += " " + nkibench.explain_roofline(intensity)
    return reward, parts, note


# ---------------------------------------------------------------- prompting

# Every name here appears in the three shipped tutorial kernels, so none of it is invented. The
# model does not know this API and guesses plausible names -- nl.scalar, nl.value, nl.dot,
# tile.mean -- none of which exist. A short card of what IS real costs ~200 tokens and is
# documentation rather than the answer.
API_CARD = """Available NKI functions:

  @nki.jit                                  decorate the entry point
  nl.ndarray(shape, dtype=..., buffer=b)    allocate; b is nl.sbuf, nl.psum or nl.shared_hbm
  nl.affine_range(n)                        the loop
  nl.sum(view, axis=[i, j])                 reduce; axis is a list
  nl.float32, nl.bfloat16                   dtypes
  nisa.dma_copy(dst=, src=)                 move data between HBM and SBUF
  nisa.nc_matmul(dst=, stationary=, moving=)   matmul into a PSUM tile
  nisa.tensor_copy(dst=, src=)              copy, e.g. PSUM to SBUF
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5)   scale by a constant
  tile.ap([[stride, count], ...])           a strided view, for reductions

nisa.nc_matmul has strict memory rules: dst must live in nl.psum, while stationary and moving must
both live in nl.sbuf. So the pattern is: dma_copy both operands from HBM into sbuf tiles, allocate a
psum tile for the result, call nc_matmul(dst=psum_tile, stationary=..., moving=...), then
tensor_copy from psum into an sbuf tile, then dma_copy that out to the shared_hbm output.
The left operand arrives already transposed, with K on the partition axis.

Slice tiles with ranges, e.g. a[0:128, 0:64]. A complete kernel looks like this:

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def copy_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    tile = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=a)
    nisa.dma_copy(dst=out, src=tile)
    return out

"""


# ---------------------------------------------------------------- experiment: what the docs say
#
# Measured in our baseline (Qwen3-8B, seat pod, greedy server): on level 1 -- average pooling, which
# needs no matmul -- the model wrote nc_matmul calls round after round. The API card is the same for
# every level, and its longest paragraph is the nc_matmul memory rules. Hypothesis: the card tells
# the model what kind of kernel to write. These flags test that, one change each, all OFF by
# default so the baseline prompt is untouched.

_MATMUL_LINE = "  nisa.nc_matmul(dst=, stationary=, moving=)   matmul into a PSUM tile\n"
_MATMUL_PARA = API_CARD[API_CARD.index("nisa.nc_matmul has strict memory rules"):
                        API_CARD.index("Slice tiles with ranges")]
_MATMUL_LIMIT = (f" For matmul, the stationary free dimension is at most "
                 f"{nkibench.GEMM_STATIONARY_FMAX} and the moving free dimension at most "
                 f"{nkibench.GEMM_MOVING_FMAX}.")


def uses_matmul(level):
    return nkibench.LEVELS[level]["ref"] in (nkibench.ref_matmul, nkibench.ref_attention)


def level_card(level):
    """--api-card level: the API card without the matmul material on levels that need no matmul."""
    if uses_matmul(level):
        return API_CARD
    return API_CARD.replace(_MATMUL_LINE, "").replace(_MATMUL_PARA, "")


# --ap-example: the strided view is the one idiom level 1 needs and the card names it in a single
# line. This is a GENERIC example (pairs along one free axis), not the pooling kernel.
AP_EXAMPLE = """A strided view groups elements so one nl.sum reduces them. For a tile t of shape (P, F),
summing each adjacent pair along the free axis:
    v = t.ap([[F, P], [2, F // 2], [1, 2]])      # [stride, count] per axis -> shape (P, F//2, 2)
    s = nl.sum(v, axis=[2])                      # shape (P, F//2)
The first [stride, count] pair is always the partition axis. Strides count elements of t.
"""

# --repair-card: repair prompts carry no API card at all, and our baseline showed the model
# inventing names (nisa.multiply) in repair rounds. When the latest failure is an invented name,
# send this short list of what is real alongside the repair.
REPAIR_CARD = """The NKI calls that exist, for reference:
  nl.ndarray(shape, dtype=..., buffer=nl.sbuf | nl.psum | nl.shared_hbm)   nl.affine_range(n)
  nl.sum(view, axis=[...])   tile.ap([[stride, count], ...])   nl.float32   nl.multiply
  nisa.dma_copy(dst=, src=)   nisa.tensor_copy(dst=, src=)
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=x)
  nisa.nc_matmul(dst=psum, stationary=sbuf, moving=sbuf)
Use only these. Anything else will not exist.
"""
_INVENTED = re.compile(r"has no attribute|is not callable|unexpected keyword argument|"
                       r"no module named", re.I)


# ---------------------------------------------------------------- experiment: name the tensor
#
# --shape-hint. Measured in our baseline: `dma_copy requires src and dst to have the same number of
# elements` was the most common failure, on levels 1, 2 and 3. On level 3 the model allocated ONE
# (128, 512) tile and copied lhsT (128, 64), then rhs, into it. The existing message only knows two
# numbers, so it offers a generic "a 128x512 piece" example -- which points AWAY from the fix, since
# lhsT has 64 columns. But the checker holds the actual inputs: 8192 elements is exactly lhsT. So
# name the tensor, its shape, and the one change: give it its own tile of its own shape.

SHAPE_HINT = False
RESHAPE_HINT = False


# --reshape-hint. Measured in exp3: level 3 ended on `cannot reshape array of size 32768 into shape
# (1,64)` four rounds running, and the model resent the same kernel each time. The kernel has no
# reshape. The error is raised inside nc_matmul: the model allocated the PSUM destination as (1, 64),
# 64 elements, and the product of lhsT (K, M) and rhs (K, N) is M x N. enrich() answers "do not
# reshape", which blames something the model never wrote. Name the real cause and the real shape.

def reshape_hint(err, args, level):
    """A matmul-destination instruction for a 'cannot reshape' error, or '' to fall back."""
    m = re.search(r"cannot reshape array of size (\d+) into shape \((\d+),\s*(\d+)\)", err)
    if not m:
        return ""
    size, r, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    import inspect
    names = list(inspect.signature(nkibench.LEVELS[level]["ref"]).parameters)
    mats = [(n, x) for n, x in zip(names, args) if isinstance(x, np.ndarray) and x.ndim == 2]
    if len(mats) < 2 or mats[0][1].shape[1] * mats[1][1].shape[1] != size:
        return ""
    (sn, sa), (mn, ma) = mats[0], mats[1]
    M, N = sa.shape[1], ma.shape[1]
    shape = f"({sn}.shape[1], {mn}.shape[1])"
    # Exp 4b: given only the numbers (64, 512), the model wrote (lhsT.shape[1], rhs.shape[0]) and
    # got (64, 128): it indexed the wrong axis of rhs. So give the expression, and name the trap.
    return (f"{err}. This is raised inside nisa.nc_matmul, not by a reshape in your code: the "
            f"destination tile `dst` you passed has shape ({r}, {c}) and holds {r * c} elements, "
            f"but this matmul produces {size} elements. The product is (columns of stationary) x "
            f"(columns of moving) = ({M}, {N}). Write the PSUM destination as "
            f"`psum = nl.ndarray({shape}, dtype=nl.float32, buffer=nl.psum)`. Use index [1] on both "
            f"inputs: `{mn}.shape[0]` is {ma.shape[0]}, the contraction length, not the {N} you "
            f"need. Give the output tile and the shared_hbm output the same {shape} shape.")


def shape_hint(err, args, level):
    """A tensor-specific instruction for a dma_copy size mismatch, or '' to fall back to enrich()."""
    m = re.search(r"same number of elements, got src=(\d+), dst=(\d+)", err)
    if not m:
        return ""
    import inspect
    src_n, dst_n = int(m.group(1)), int(m.group(2))
    names = list(inspect.signature(nkibench.LEVELS[level]["ref"]).parameters)
    tensors = [(n, a) for n, a in zip(names, args) if isinstance(a, np.ndarray)]
    for name, a in tensors:
        if a.size == src_n:
            head = (f"{err}. You copied the whole input `{name}` (shape {a.shape}, {src_n} "
                    f"elements) into a tile that holds {dst_n} elements. Each tensor needs its OWN "
                    f"tile with exactly its own shape -- never reuse one tile for two different "
                    f"tensors.")
            if a.shape[0] > nkibench.PMAX:
                rest = ", ".join(str(d) for d in a.shape[1:])
                return (head + f" `{name}` has {a.shape[0]} rows and a tile holds at most "
                        f"{nkibench.PMAX}, so load it in pieces: inside a loop over "
                        f"nl.affine_range({a.shape[0]} // {nkibench.PMAX}), allocate "
                        f"`t = nl.ndarray(({nkibench.PMAX}, {rest}), dtype={name}.dtype, "
                        f"buffer=nl.sbuf)` and copy `nisa.dma_copy(dst=t, "
                        f"src={name}[i*{nkibench.PMAX}:(i+1)*{nkibench.PMAX}, :])`.")
            return (head + f" Write `{name}_tile = nl.ndarray({name}.shape, dtype={name}.dtype, "
                    f"buffer=nl.sbuf)` and then `nisa.dma_copy(dst={name}_tile, src={name})`, and do "
                    f"the same, separately, for every other input.")
    try:
        out = nkibench.LEVELS[level]["ref"](*args)
    except Exception:
        out = None
    if out is not None and np.size(out) == dst_n:
        return (f"{err}. The destination is the whole output (shape {np.shape(out)}, {dst_n} "
                f"elements), but the tile you copied out of holds {src_n}. The tile you copy into "
                f"the output must have the output's exact shape {np.shape(out)} -- or copy into the "
                f"matching slice of the output instead of the whole thing.")
    return ""


def available_names(dotted):
    """Turn 'no attribute X' into 'here are the real ones'.

    Measured: told only `module 'nki.language' has no attribute 'value'`, the model guessed
    another invented name every round -- value, scalar, sbuf_scalar, dot. A verdict names the
    mistake and never the fix, so list what actually exists and let it choose.
    """
    import difflib
    import importlib
    mod_name, _, attr = dotted.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return ""
    names = [n for n in dir(mod) if not n.startswith("_")]
    close = difflib.get_close_matches(attr, names, n=6, cutoff=0.4)
    if close:
        return (f" `{mod_name}` has no `{attr}`. The closest real names are: "
                f"{', '.join(close)}. Pick one of those or use a different approach.")
    return (f" `{mod_name}` has no `{attr}`, and nothing similar exists. Its real names include: "
            f"{', '.join(sorted(names)[:25])}.")


def real_signature(func_name):
    """The actual signature of an NKI function, for when the model invents arguments."""
    import inspect
    for mod_name in ("nki.language", "nki.isa", "nki"):
        try:
            mod = __import__(mod_name, fromlist=["x"])
        except Exception:
            continue
        fn = getattr(mod, func_name, None)
        if fn is None:
            continue
        try:
            return f"{mod_name.split('.')[-1]}.{func_name}{inspect.signature(fn)}"
        except (TypeError, ValueError):
            return f"{mod_name.split('.')[-1]}.{func_name}"
    return ""


def enrich(error_text):
    """Add the real names when the failure is an invented API call."""
    if "'MemoryRegion' object is not callable" in error_text:
        return (error_text + " nl.sbuf, nl.psum and nl.shared_hbm are memory regions, not "
                "functions. Do not call them. Allocate with "
                "nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf) and pass the region as the "
                "buffer= argument.")
    # Added after our experiment 2: once the model stopped inventing names it reached the right
    # call, nisa.tensor_scalar, and left arguments out. The raw TypeError named the missing argument
    # but not the call's shape, so it fixed one and dropped another (operand0, then data).
    m = re.search(r"(\w+)\(\) missing \d+ required (?:positional|keyword-only) arguments?: (.+)",
                  error_text)
    if m:
        sig = real_signature(m.group(1))
        return (error_text + f" Supply every required argument, by keyword."
                + (f" The real signature is {sig}." if sig else "")
                + (" For scaling a tile by a constant: nisa.tensor_scalar(dst=out_tile, "
                   "data=in_tile, op0=nl.multiply, operand0=0.25)."
                   if m.group(1) == "tensor_scalar" else ""))
    # Also from experiment 2: a PSUM tile copied straight to the HBM output.
    if re.search(r"dma_copy requires HBM or SBUF tensors, got src=MemoryRegion\.psum", error_text):
        return (error_text + " nisa.dma_copy cannot read PSUM. Copy the PSUM tile into an SBUF tile "
                "of the same shape with nisa.tensor_copy(dst=sbuf_tile, src=psum_tile) first, then "
                "nisa.dma_copy(dst=<the shared_hbm output>, src=sbuf_tile).")
    m = re.search(r"(\w+)\(\) got an unexpected keyword argument '(\w+)'", error_text)
    if m:
        sig = real_signature(m.group(1))
        return (error_text + f" Remove the `{m.group(2)}=` argument."
                + (f" The real signature is {sig}." if sig else ""))
    if "unsupported operand type(s) for" in error_text and "NkiTensor" in error_text:
        return (error_text + " A tile is not a number, so Python operators like += do not work on "
                "one. Accumulate by allocating a PSUM tile with "
                "nl.ndarray(shape, nl.float32, buffer=nl.psum) and letting nisa.nc_matmul add into "
                "it across the loop, or combine two tiles with a nisa op rather than a Python "
                "operator.")
    m = re.search(r"dma_copy requires src and dst to have the same number of elements, "
                  r"got src=(\d+), dst=(\d+)", error_text)
    if m:
        src, dst = int(m.group(1)), int(m.group(2))
        return (error_text + f" The tile you allocated holds {dst} elements but you copied {src} "
                f"into it. nisa.dma_copy does not slice or broadcast: allocate the destination with "
                f"EXACTLY the shape of the slice you are moving. If you want a 128x512 piece of a "
                f"bigger tensor, write "
                f"t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then "
                f"nisa.dma_copy(dst=t, src=a[0:128, 0:512]) -- the slice on the right must have the "
                f"same shape as the tile on the left.")
    m = re.search(r"dma_copy (\w+) partition dimension (\d+) exceeds maximum (\d+)", error_text)
    if m:
        which, got, mx = m.group(1), int(m.group(2)), int(m.group(3))
        return (error_text + f" A tile may have at most {mx} rows, and you asked for {got}. Do not "
                f"allocate one tile for the whole tensor: loop over the partition dimension in "
                f"chunks of at most {mx} with nl.affine_range, allocate the tile inside the loop with "
                f"the chunk's own size, and copy one chunk at a time, e.g. "
                f"src=a[i*{mx}:(i+1)*{mx}, :]. If a dimension is already {mx} or smaller, use it "
                f"whole -- do NOT pad it up to {mx}, that reads past the end of the tensor. The same "
                f"applies to where you write the result back.")
    m = re.search(r"value array of shape \((\d+),?\) could not be broadcast to "
                  r"indexing result of shape \((\d+),?\)", error_text)
    if m:
        val, dst = int(m.group(1)), int(m.group(2))
        return (error_text + f" You assigned {val} elements into a slice that holds {dst}. Assignment "
                f"does not reshape or broadcast either: the slice on the left and the value on the "
                f"right must have the SAME shape. If the value is bigger, you are writing a whole tile "
                f"where a slice belongs -- index the destination to match, e.g. "
                f"out[i*128:(i+1)*128, :] = tile. If it is smaller, you are looping over the wrong "
                f"dimension.")
    m = re.search(r"Out-of-bound access for tensor .*? on dimension (\d+): "
                  r"index range \[(\d+), (\d+)\] exceed dimension size of (\d+)",
                  error_text)
    if m:
        dim, hi, size = m.group(1), int(m.group(3)), int(m.group(4))
        return (error_text + f" You indexed up to {hi} on dimension {dim}, which is only {size} "
                f"long. Tile limits are a MAXIMUM, not a target. Derive every bound from the tensor's "
                f"own shape -- use min(limit, size) and let the final chunk be partial -- rather than "
                f"writing a fixed number. Note the two limits differ: the partition dimension (first) "
                f"allows at most 128, the free dimension allows more.")
    m = re.search(r"Matmul contraction dimension (\d+) exceeds pmax=(\d+)", error_text)
    if m:
        k, mx = int(m.group(1)), int(m.group(2))
        return (error_text + f" The contraction dimension K is {k} and one nc_matmul can only "
                f"contract {mx}. Split K into chunks of {mx} and accumulate: allocate ONE psum tile "
                f"OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so "
                f"the partial products add up there, and only after the loop copy it out with "
                f"nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write partial "
                f"results to HBM.")
    m = re.search(r"(\w+) (?:dst|src)? ?must be in \['sbuf', 'psum'\], got shared_hbm", error_text)
    if m:
        return (error_text + f" `nisa.{m.group(1)}` only moves data between on-chip buffers, sbuf and "
                f"psum. To reach HBM -- the tensor you allocated with buffer=nl.shared_hbm and will "
                f"return -- use nisa.dma_copy instead. The usual sequence is nc_matmul into psum, "
                f"tensor_copy psum to sbuf, then dma_copy sbuf to the shared_hbm output.")
    m = re.search(r"(\w+) must be in \['(\w+)'\], got (\w+)", error_text)
    if m:
        which, needed, got = m.groups()
        place = {"psum": "nl.psum", "sbuf": "nl.sbuf"}.get(needed, needed)
        return (error_text + f" Allocate the `{which}` tile with buffer={place} instead of "
                f"nl.{got}. For nisa.nc_matmul: dst must be in nl.psum, and stationary and moving "
                f"must both be in nl.sbuf. Copy between them with nisa.tensor_copy.")
    if "got multiple values for argument" in error_text:
        return (error_text + " Pass every argument by keyword, e.g. "
                "nisa.nc_matmul(dst=..., stationary=..., moving=...), so none is bound twice.")
    if "must have at least 2 dimensions" in error_text:
        return (error_text + " Every SBUF and PSUM tile needs two dimensions: a partition dimension "
                "first, then a free dimension. A 1-D tile is not allowed, so write "
                "nl.ndarray((rows, cols), ...) and give a length-N vector the shape (1, N) or "
                "(N, 1) depending on which axis you are reducing over.")
    if "cannot reshape array of size" in error_text:
        return (error_text + " Do not reshape. Work with the shapes you were given and slice "
                "them into tiles, e.g. src=a[0:128, 0:64].")
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", error_text)
    if m:
        return error_text + available_names(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_text)
    if m:
        return (error_text + f" A {m.group(1)} is not a numpy array, so it has no "
                f"`{m.group(2)}`. Use the nl/nisa functions instead.")
    return error_text

# ---------------------------------------------------------------- prompt parts
#
# Every prompt is built as a list of (part, text) pairs and joined only when sent. The joined text
# is byte-identical to what the original string-building code produced, so the agent's behaviour
# is unchanged; the parts exist so each attempt can log where its input budget went. The challenge
# names that breakdown -- docs vs error context vs code, per attempt -- as the artifact to bring.
#
# Part names, chosen so they sum to the whole prompt:
#   instructions   the task framing and the "reply with one code block" tail
#   reference      the NumPy reference implementation
#   docs           hardware limits, the API card, the matmul memory rules
#   kernel         the previous kernel, sent back for repair
#   feedback       the checker's message (after enrich)
#   ledger         the list of approaches that already failed

def join_parts(parts):
    return "".join(text for _, text in parts)


def first_prompt(level, terse=0, card="all", ap_example=False):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.

    Returns a list of (part, text) pairs; join_parts() gives the prompt.
    """
    s = nkibench.LEVELS[level]
    import inspect
    ref_src = inspect.getsource(s['ref'])
    if terse >= 2:
        # Last resort. Measured on this endpoint: one-sentence prompts answered in 300-700
        # tokens while every structured, rule-carrying prompt spiralled.
        return [
            ("instructions", f"Write a Python function `{s['entry']}` decorated with @nki.jit that "
                             f"computes the same thing as this, using nki.language as nl and "
                             f"nki.isa as nisa:\n\n"),
            ("reference", ref_src),
            ("instructions", "\nReply with one python code block."),
        ]
    if terse >= 1:
        # The matmul memory rules are the substance of levels 3 and 4, and the short prompt has to
        # carry them: measured, the agent cycled between "dst must be in ['psum']" and "moving must
        # be in ['sbuf']" because nothing told it where the operands live.
        mm = ("nisa.nc_matmul(dst=, stationary=, moving=) needs dst in nl.psum and both operands "
              "in nl.sbuf. So: dma_copy the operands HBM->sbuf, allocate a psum tile, nc_matmul "
              "into it, tensor_copy psum->sbuf, then dma_copy sbuf->the shared_hbm output you "
              "return. The left operand is already transposed, with K on the partition axis.\n"
              if level >= 3 else "")
        return [
            ("instructions", f"Write an AWS Neuron NKI kernel: a function `{s['entry']}` decorated "
                             f"with @nki.jit that computes what this reference computes.\n\n"),
            ("reference", ref_src),
            ("docs", f"\nAllocate with nl.ndarray(shape, dtype=..., buffer=nl.sbuf), move data with "
                     f"nisa.dma_copy(dst=, src=), loop with nl.affine_range(n). A tile's partition "
                     f"dimension is at most {nkibench.PMAX}.\n{mm}"),
            ("instructions", "\nReply with one python code block."),
        ]
    trim = card == "level" and not uses_matmul(level)
    card_text = level_card(level) if card == "level" else API_CARD
    if ap_example and not uses_matmul(level):
        card_text = card_text + "\n" + AP_EXAMPLE
    return [
        ("instructions", f"Write an AWS Neuron NKI kernel.\n\n"
                         f"Operation: {s['op']}\n"
                         f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
                         f"It must compute exactly what this NumPy reference computes:\n\n"),
        ("reference", ref_src),
        ("docs", f"\nHardware limits: a tile's partition dimension is at most {nkibench.PMAX}."
                 + ("" if trim else _MATMUL_LIMIT) + "\n\n"
                 f"Import nki, nki.language as nl, and nki.isa as nisa.\n\n{card_text}\n\n"),
        ("instructions", "Reply with ONE python code block containing the imports and the "
                         "function. No prose."),
    ]


def repair_prompt(level, source, feedback, repair_card=False):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.

    Returns a list of (part, text) pairs; join_parts() gives the prompt.
    """
    return [
        ("instructions", f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"),
        ("kernel", f"```python\n{source}\n```\n\n"),
        ("instructions", "A checker reports:\n"),
        ("feedback", f"{feedback}"),
        ("instructions", "\n\nChange exactly what the checker names and keep everything else "
                         "identical. Reply with ONE python code block."),
    ] + ([("docs", "\n\n" + REPAIR_CARD)] if repair_card and _INVENTED.search(feedback or "")
         else [])


CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text):
    """Pull out the code, and return NOTHING rather than prose.

    The old fallback returned the whole reply whenever it contained "def ", so a numbered list
    or a sentence reached the compiler and produced "invalid decimal literal on line 2" -- a
    parse error that blamed the model for the extractor's mistake.
    """
    text = text or ""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    # No fence: start at the first line that can legally begin a module and keep the rest.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |@nki|def )", line):
            return "\n".join(lines[i:]).strip()
    return ""


# ---------------------------------------------------------------- the model

def ask(a, prompt):
    """Returns the answer text. ask_meta() returns the same call's usage numbers too."""
    return ask_meta(a, prompt)["content"]


def ask_meta(a, prompt):
    import httpx
    t0 = time.perf_counter()
    # enable_thinking=False matters. Qwen3 reasons before answering, and with thinking on it
    # spent the whole budget there: the first cluster run returned "No code came back" at 54.7s
    # over and over, plus truncated fragments (invalid decimal literal, unterminated string).
    # Keep prompt + answer inside the server's context, or the answer is silently cut off and
    # every parse error below is really a budget error. Repair prompts grow with the kernel.
    est_prompt = len(prompt) // 4
    budget = min(a.max_tokens, max(256, a.context - est_prompt - 64))
    if budget < a.max_tokens:
        print(f"    (prompt is ~{est_prompt} tokens, so the answer budget is capped at {budget} "
              f"to stay inside the {a.context}-token context)")
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    reasoning = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    content = msg.get("content") or ""
    finish = ch.get("finish_reason")
    if finish == "length":
        # Do not let a budget problem look like a model failure.
        print(f"    (TRUNCATED: finish_reason=length after {len(content)} chars. The answer was "
              f"cut off, so any parse error below is the budget, not the model. Prompt is "
              f"{len(prompt)} chars; server context is the ceiling.)")
    if not content.strip() and reasoning:
        # The single most common surprise on this endpoint, so name it rather than reporting
        # an empty answer as a model failure.
        print(f"    (empty answer, {len(reasoning)} chars of hidden reasoning, "
              f"finish={ch.get('finish_reason')} — shorten the prompt rather than raising the "
              f"budget)")
    usage = payload.get("usage") or {}
    return dict(content=content, finish=finish, reasoning_chars=len(reasoning),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                answer_budget=budget, seconds=round(time.perf_counter() - t0, 2))


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask_meta, a, prompt) for _ in range(n)]]


# ---------------------------------------------------------------- token accounting

_TOKENIZE_URL = None      # resolved on first use; False once the server has refused
_TOKEN_CACHE = {}


def count_tokens(a, text):
    """Exact token count from the server's /tokenize endpoint, or len/4 if it has none.

    vLLM serves /tokenize next to /v1, so this asks the same model's own tokenizer -- no local
    tokenizer, no guessing. The shared gpt-oss endpoint may not expose it; one failure switches
    to the estimate for the rest of the run, and the log records which method was used.
    Returns (tokens, method).
    """
    global _TOKENIZE_URL
    if not text:
        return 0, "exact"
    if text in _TOKEN_CACHE:
        return _TOKEN_CACHE[text]
    if _TOKENIZE_URL is None:
        base = (a.base or "").rstrip("/")
        _TOKENIZE_URL = (base[:-3] if base.endswith("/v1") else base) + "/tokenize" \
            if (base and not a.offline and a.token_count == "server") else False
    if _TOKENIZE_URL:
        try:
            import httpx
            r = httpx.post(_TOKENIZE_URL, json=dict(model=a.model, prompt=text,
                                                    add_special_tokens=False),
                           timeout=30, verify=False)
            r.raise_for_status()
            got = (int(r.json()["count"]), "exact")
            _TOKEN_CACHE[text] = got
            return got
        except Exception as e:
            print(f"    (token counting: {_TOKENIZE_URL} unavailable ({type(e).__name__}); "
                  f"falling back to chars/4 for the rest of this run)")
            _TOKENIZE_URL = False
    return max(1, len(text) // 4), "estimate"


def account(a, parts):
    """Tokens per part name, summed over repeated names. ({part: tokens}, method)."""
    out, methods = {}, set()
    for name, text in parts:
        n, m = count_tokens(a, text)
        out[name] = out.get(name, 0) + n
        methods.add(m)
    return out, ("exact" if methods == {"exact"} else "estimate")


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    text = (f"```python\n{ref.replace('@nki.jit', '', 1)}\n```" if rnd == 0
            else f"```python\n{ref}\n```")
    return [dict(content=text, finish="offline", reasoning_chars=0, prompt_tokens=None,
                 completion_tokens=None, answer_budget=None, seconds=0.0)] * n


# ---------------------------------------------------------------- the loop

def verdict(level, best_reward, best_feedback, solved):
    """What the agent claims about a level, stated so it can be checked.

    The scoring rubric punishes "verified" on a kernel that fails, and rewards an agent that says
    plainly what it could not verify. So the claim is derived only from the checker's own result,
    and it says which checker: layer 1 is the CPU simulator, never the device.
    """
    n = len(nkibench.LEVELS[level]["shapes"])
    if solved:
        return ("VERIFIED", f"correct on all {n} test shapes in nki.simulate (CPU), no hardware "
                            f"hazard warnings. Not run or timed on the device.")
    m = re.match(r"(\d+) of (\d+) shapes passed", best_feedback or "")
    passed = f"{m.group(1)} of {m.group(2)} shapes passed" if m else "no shape passed"
    return ("NOT VERIFIED", f"best reward {best_reward:.2f}; {passed}. Do not use this kernel.")


def solve(a, level, log, run=0):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt_parts = first_prompt(level, terse, a.api_card, a.ap_example)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")

    def finish(reward, rounds, solved, why):
        status, detail = verdict(level, best[0], best[2], solved)
        print(f"  VERDICT: {status} -- {detail}")
        log.write(json.dumps(dict(type="result", run=run, level=level, best=best[0],
                                  rounds=rounds, solved=solved, verdict=status, detail=detail,
                                  stopped=why, model=a.model, samples=a.samples,
                                  context=a.context, terse=a.terse, api_card=a.api_card,
                                  ap_example=a.ap_example, repair_card=a.repair_card,
                                  shape_hint=a.shape_hint,
                                  reshape_hint=a.reshape_hint)) + "\n")
        log.flush()
        return reward, rounds

    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        prompt = join_parts(prompt_parts)
        budget_by_part, method = account(a, prompt_parts)
        kind = ("first" if rnd == 0 or not any(n == "kernel" for n, _ in prompt_parts)
                else "repair+ledger" if any(n == "ledger" for n, _ in prompt_parts) else "repair")
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, prompt, a.samples))
        graded = []
        for sample, meta in enumerate(replies):
            reply = meta["content"] or ""
            src = extract_code(reply)
            reward, parts, feedback = grade(src, level)
            graded.append((reward, src, feedback, parts))
            log.write(json.dumps(dict(type="attempt", run=run, level=level, round=rnd,
                                      sample=sample, reward=reward, parts=parts,
                                      prompt_kind=kind, terse=terse,
                                      prompt_chars=len(prompt), reply_chars=len(reply),
                                      prompt_parts=budget_by_part, token_method=method,
                                      prompt_tokens=meta["prompt_tokens"],
                                      completion_tokens=meta["completion_tokens"],
                                      answer_budget=meta["answer_budget"],
                                      finish=meta["finish"],
                                      reasoning_chars=meta["reasoning_chars"],
                                      seconds=meta["seconds"],
                                      code=src, feedback=feedback)) + "\n")
        log.flush()
        spent = ", ".join(f"{k} {v}" for k, v in budget_by_part.items())
        graded.sort(key=lambda g: g[0], reverse=True)
        top = graded[0]
        if top[0] > best[0]:
            best = (top[0], top[1], top[2])
        # Repair the LATEST attempt, not the best one. Rebuilding from the best attempt with the
        # best attempt's feedback is a fixed point: once a round scores worse, the prompt stops
        # changing, and a greedy model then returns the same answer forever. Measured: level 2
        # stuck at 0.10 for four rounds while the prompt still carried the 0.50 code.
        if (top[1] or "").strip():
            latest = (top[1], top[2])
        same = top[2] == (tried[-1] if tried else None)
        if same:
            # Collapse. Fifteen identical multi-line blocks is noise, not information.
            print(f"round {rnd}: same failure again ({top[0]:.2f}, best so far {best[0]:.2f})"
                  f"  [input tokens: {spent}]")
        else:
            print(f"round {rnd}: this round {top[0]:.2f}  best so far {best[0]:.2f}  "
                  f"({time.perf_counter() - t0:.1f}s)  [input tokens: {spent}]")
            print(f"  {top[2][:400]}")
        if top[0] >= sum(WEIGHTS.values()) - 1e-9:
            print(f"  SOLVED on round {rnd}. {top[2]}")
            print("  ---------------- the kernel ----------------")
            print(textwrap.indent(top[1], "  "))
            print("  -------------------------------------------")
            return finish(top[0], rnd + 1, True, "solved")
        seen[top[2]] = seen.get(top[2], 0) + 1
        streak = streak + 1 if same else 1
        if seen[top[2]] >= a.give_up_after:
            how = ("the identical failure %d rounds running" % streak if streak >= a.give_up_after
                   else "this failure for the %dth time, alternating with %d other(s)"
                        % (seen[top[2]], len(seen) - 1))
            print(f"  STOPPING this level: {how}. The agent is cycling between a fixed set of "
                  f"mistakes rather than converging, so more rounds will not help. Failures seen:")
            for f, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"    {n}x  {f[:110]}")
            return finish(best[0], rnd + 1, False, "cycling")
        tried.append(top[2])
        repeats = streak
        if repeats >= 2 and (best[1] or "").strip():
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt_parts = (repair_prompt(level, latest[0], latest[1], a.repair_card)
                            + [("instructions", "\n\nThese approaches have already failed, so do "
                                                "something different:\n"),
                               ("ledger", ledger)])
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt_parts = first_prompt(level, terse, a.api_card, a.ap_example)
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt_parts = repair_prompt(level, latest[0], latest[1], a.repair_card)
    print(f"  not solved in {a.rounds} rounds; best reward {best[0]:.2f}")
    return finish(best[0], a.rounds, False, "out of rounds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    ap.add_argument("--all", action="store_true", help="levels 1 to 4")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=MIN_ANSWER_TOKENS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL")
                    or os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", default="", help="path to append, e.g. /agg/v1 for gpt-oss")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the whole thing N times and report a solve RATE. One run is not a "
                         "result: measured, the same config scored 1.00, 1.00 and 0.50 on level 2 "
                         "across three runs with no code change.")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--give-up-after", type=int, default=4,
                    help="stop a level after this many identical failures in a row. Measured: 15 "
                         "was pure waste, because the prompt had stopped changing.")
    ap.add_argument("--terse", type=int, default=0, choices=(0, 1, 2),
                    help="starting prompt length. Measured on gpt-oss-20b: 0 produced 13,245 "
                         "chars of hidden reasoning and no answer, while 1 answered with code. "
                         "Qwen3-8B is fine at 0.")
    ap.add_argument("--context", type=int, default=4096,
                    help="the server's max-model-len; prompt + answer must fit inside it")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first; costs budget, and it ran out")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--token-count", default="server", choices=("server", "estimate"),
                    help="how to count input tokens per prompt part for the log: 'server' asks the "
                         "model server's /tokenize (exact, falls back on failure), 'estimate' is "
                         "chars/4. Only the log is affected, never the prompt.")
    ap.add_argument("--api-card", default="all", choices=("all", "level"),
                    help="EXPERIMENT. 'all' (default, the baseline) sends the full API card on "
                         "every level; 'level' drops the nc_matmul material on levels that need "
                         "no matmul (1 and 2).")
    ap.add_argument("--ap-example", action="store_true",
                    help="EXPERIMENT. Add a generic strided-view + nl.sum example to the card on "
                         "non-matmul levels.")
    ap.add_argument("--repair-card", action="store_true",
                    help="EXPERIMENT. When the latest failure is an invented API name, include a "
                         "short list of the real calls in the repair prompt.")
    ap.add_argument("--shape-hint", action="store_true",
                    help="EXPERIMENT. On a dma_copy size mismatch, name the input tensor whose size "
                         "matches and tell the model to give it its own tile of its own shape.")
    ap.add_argument("--reshape-hint", action="store_true",
                    help="EXPERIMENT. On a 'cannot reshape' error from nc_matmul, name the real "
                         "cause (wrong destination tile shape) and give the (M, N) shape to use.")
    a = ap.parse_args()
    global SHAPE_HINT, RESHAPE_HINT
    SHAPE_HINT = a.shape_hint
    RESHAPE_HINT = a.reshape_hint

    if not a.offline:
        # Validate before the first request. An empty or scheme-less value produces a hostname
        # starting with "." and 30 lines of httpx/idna traceback that say nothing about the cause.
        raw = (a.base or "").strip()
        if not raw:
            sys.exit("KERNEL_AGENT_BASE_URL is empty or unset. Point it at a model:\n"
                     "  export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1\n"
                     "  export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B\n"
                     "That is the server ./serve.sh started. The seat pods set these for you; if they are "
                     "missing, you are not in a seat pod. Or pass --offline to run with no model.\n"
                     "(Separately: the shared gpt-oss endpoint is reached by setting "
                     "GPTOSS_BASE_URL and passing --path /agg/v1 -- not needed for the local run.)")
        from urllib.parse import urlparse
        u = urlparse(raw)
        if u.scheme not in ("http", "https") or not u.netloc:
            sys.exit(f"base URL {raw!r} is not usable. It needs a scheme and a host, e.g. "
                     f"http://qwen3-8b:8000/v1.")
        a.base = raw.rstrip("/") + a.path
        print(f"endpoint {a.base}  model {a.model}")
    else:
        print("*** OFFLINE: replaying the reference kernel. Numbers are meaningless. ***")

    levels = sorted(nkibench.LEVELS)[:4] if a.all else [a.level or 1]
    full = sum(WEIGHTS.values())
    history = {lv: [] for lv in levels}

    with open(a.log, "a") as log:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            results = []
            for level in levels:
                results.append((level,) + solve(a, level, log, run=rep))
                history[level].append(results[-1][1])

            print("\n=========== summary ===========")
            for level, reward, rounds in results:
                print(f"  level {level}  reward {reward:.2f} after {rounds} round(s)"
                      + ("  SOLVED" if reward >= full - 1e-9 else ""))
            print(f"  solved {sum(1 for _, r, _ in results if r >= full - 1e-9)}/{len(results)}")

    if a.repeat > 1:
        # The number that actually means something. A solve rate over N runs survives the variance
        # that makes any single run uninterpretable.
        print(f"\n=========== over {a.repeat} runs ===========")
        for lv in levels:
            got = history[lv]
            solves = sum(1 for r in got if r >= full - 1e-9)
            print(f"  level {lv}: solved {solves}/{len(got)}  "
                  f"best {max(got):.2f}  worst {min(got):.2f}  "
                  f"mean {sum(got) / len(got):.2f}  all={[round(r, 2) for r in got]}")
        print("\n  Report the rate, not your best run. A level that solves 1 in 3 times is not solved.")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
