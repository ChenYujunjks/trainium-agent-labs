#!/usr/bin/env python3
"""
analyze.py -- turn attempt logs into the three things the judges ask for.

    python analyze.py baseline.jsonl                     one log
    python analyze.py baseline.jsonl feedback_v1.jsonl   compare two, side by side
    python analyze.py run.jsonl --out report/            also write CSVs (and PNGs if matplotlib exists)

What it reports, per log:

  1. Solve RATE per level -- runs solved / runs, with best, worst, mean. One run is not a result.
  2. FAILURE TAXONOMY -- every attempt's checker message classified into a named failure mode,
     counted per level, plus the failure each run was stuck on when it ended.
  3. TOKEN BUDGET -- input tokens per round, split into instructions / reference / docs / kernel /
     feedback / ledger. Only logs written by the instrumented agent.py carry this.
  4. CALIBRATION -- whether every "VERIFIED" claim the agent made is backed by a 1.0 reward.

Reads both log formats: the original agent.py (no `type`/`run` fields; runs are inferred from the
round counter resetting) and the instrumented one.
"""

import argparse
import collections
import csv
import json
import os
import re
import sys

FULL = 1.0

# Ordered: the first pattern that matches names the failure. Each name is a failure MODE, i.e. a
# thing the model misunderstood, not a Python exception type -- two different exceptions with the
# same cause belong together, and that grouping is the analysis.
TAXONOMY = [
    ("no_code",            r"No code came back"),
    ("does_not_parse",     r"does not parse"),
    ("rule_violation",     r"Rule violations"),
    ("invented_module",    r"There is no module named|could not be loaded"),
    ("invented_api",       r"has no attribute|is not callable|unexpected keyword argument"),
    ("tile_1d",            r"must have at least 2 dimensions"),
    ("tile_too_big",       r"partition dimension \d+ exceeds|contraction dimension \d+ exceeds"),
    ("reshape_not_slice",  r"cannot reshape array"),
    ("shape_mismatch_copy", r"same number of elements|could not be broadcast"),
    ("out_of_bounds",      r"Out-of-bound access"),
    ("wrong_buffer",       r"must be in \[|got shared_hbm"),
    ("python_op_on_tile",  r"unsupported operand type"),
    ("input_mutated",      r"MODIFIED ITS INPUT"),
    ("hardware_hazard",    r"WRONG ON HARDWARE"),
    ("wrong_shape",        r"WRONG SHAPE"),
    ("non_finite",         r"NON-FINITE OUTPUT"),
    ("output_not_written", r"OUTPUT IS \d+% ZEROS|of the output is zero"),
    ("numeric_ragged_edge", r"NUMERICAL MISMATCH(?s:.*)ragged edge"),
    ("numeric_core",       r"NUMERICAL MISMATCH"),
    ("too_much_traffic",   r"TOO MUCH HBM TRAFFIC"),
    ("other_exception",    r"raised \w+"),
]
_COMPILED = [(n, re.compile(p)) for n, p in TAXONOMY]


def classify(feedback, reward):
    if reward >= FULL - 1e-9:
        return "solved"
    for name, rx in _COMPILED:
        if rx.search(feedback or ""):
            return name
    return "unclassified"


def load(path):
    """Attempts and results from one log. Infers run numbers for logs that do not carry them."""
    attempts, results = [], []
    run_of_level, last_round, prev_level = collections.Counter(), {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "result":
                results.append(r)
                continue
            lv, rnd = r["level"], r["round"]
            if "run" not in r:
                # Old format has no run number. A new run of a level began if its round counter
                # went backwards, or if we come back to it at round 0 after other levels ran
                # (levels run in sequence inside one run, so a revisit means the next run).
                if lv in last_round and (rnd < last_round[lv]
                                         or (rnd == 0 and prev_level not in (None, lv))):
                    run_of_level[lv] += 1
                r["run"] = run_of_level[lv]
            last_round[lv] = rnd
            prev_level = lv
            r["mode"] = classify(r.get("feedback", ""), r["reward"])
            attempts.append(r)
    return attempts, results


def per_run(attempts):
    """{(level, run): dict(best, rounds, solved, stuck_on)} from the attempts alone."""
    out = {}
    for a in attempts:
        k = (a["level"], a["run"])
        s = out.setdefault(k, dict(best=0.0, rounds=0, solved=False, last_best_mode=None,
                                   _round_best={}))
        s["best"] = max(s["best"], a["reward"])
        s["rounds"] = max(s["rounds"], a["round"] + 1)
        s["solved"] = s["solved"] or a["reward"] >= FULL - 1e-9
        rb = s["_round_best"]
        if a["round"] not in rb or a["reward"] > rb[a["round"]][0]:
            rb[a["round"]] = (a["reward"], a["mode"])
    for s in out.values():
        rb = s.pop("_round_best")
        s["stuck_on"] = rb[max(rb)][1] if rb else None
    return out


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def report(path, out_dir=None):
    attempts, results = load(path)
    runs = per_run(attempts)
    levels = sorted({lv for lv, _ in runs})
    print(f"\n#################### {path} ####################")
    print(f"{len(attempts)} attempts, {len({r for _, r in runs})} run(s), levels {levels}")

    section("1. Solve rate per level (report this, not the best run)")
    rate_rows = []
    for lv in levels:
        rs = [runs[k] for k in sorted(runs) if k[0] == lv]
        bests = [r["best"] for r in rs]
        solved = sum(r["solved"] for r in rs)
        rounds = [r["rounds"] for r in rs if r["solved"]]
        spread = "zero spread -- a capability wall" if len(set(bests)) == 1 and len(rs) > 1 \
            else ("" if len(rs) == 1 else f"spread {max(bests) - min(bests):.2f} -- luck matters here")
        print(f"  level {lv}: solved {solved}/{len(rs)}  best {max(bests):.2f}  "
              f"worst {min(bests):.2f}  mean {sum(bests) / len(bests):.2f}  "
              f"all={[round(b, 2) for b in bests]}"
              + (f"  rounds-to-solve {rounds}" if rounds else "") + (f"  ({spread})" if spread else ""))
        rate_rows.append(dict(level=lv, runs=len(rs), solved=solved, best=max(bests),
                              worst=min(bests), mean=round(sum(bests) / len(bests), 3)))

    section("2. Failure taxonomy (every attempt, by failure mode)")
    by_level = {lv: collections.Counter(a["mode"] for a in attempts if a["level"] == lv)
                for lv in levels}
    modes = [m for m, _ in collections.Counter(a["mode"] for a in attempts).most_common()]
    width = max(len(m) for m in modes) if modes else 10
    print("  " + "mode".ljust(width) + "".join(f"  L{lv:<4}" for lv in levels) + "   total")
    tax_rows = []
    for m in modes:
        counts = [by_level[lv][m] for lv in levels]
        print("  " + m.ljust(width) + "".join(f"  {c:<5}" for c in counts) + f"   {sum(counts)}")
        tax_rows.append(dict(mode=m, **{f"L{lv}": c for lv, c in zip(levels, counts)},
                             total=sum(counts)))
    print("\n  Where each run ended up (the failure of its best attempt in its final round):")
    for lv in levels:
        ends = collections.Counter(runs[k]["stuck_on"] for k in runs if k[0] == lv)
        print(f"    level {lv}: " + ", ".join(f"{m} x{n}" for m, n in ends.most_common()))

    section("3. Input token budget per round (where the context went)")
    tok = [a for a in attempts if a.get("prompt_parts")]
    tok_rows = []
    if not tok:
        print("  this log predates the instrumented agent.py, so it has no per-part token counts.\n"
              "  prompt_chars is all it has: " + ", ".join(
                  f"L{lv} first prompt {next((a['prompt_chars'] for a in attempts if a['level'] == lv), 0)} chars"
                  for lv in levels))
    else:
        parts = ["instructions", "reference", "docs", "kernel", "feedback", "ledger"]
        method = collections.Counter(a.get("token_method") for a in tok).most_common(1)[0][0]
        print(f"  token counts are {method}" + (" (server tokenizer)" if method == "exact" else
                                                " (chars/4)"))
        print("  " + "level run rnd kind          ".ljust(30)
              + "".join(p[:7].rjust(8) for p in parts) + "   total  served  fin")
        seen = set()
        for a in tok:
            k = (a["level"], a["run"], a["round"])
            if k in seen:
                continue
            seen.add(k)
            pp = a["prompt_parts"]
            total = sum(pp.values())
            print(f"  {a['level']:<5} {a['run']:<3} {a['round']:<3} {a.get('prompt_kind', ''):<13}"
                  + "".join(str(pp.get(p, 0)).rjust(8) for p in parts)
                  + f"{total:>8}{str(a.get('prompt_tokens') or ''):>8}  {a.get('finish', '')}")
            tok_rows.append(dict(level=a["level"], run=a["run"], round=a["round"],
                                 kind=a.get("prompt_kind"), **{p: pp.get(p, 0) for p in parts},
                                 total=total, served_prompt_tokens=a.get("prompt_tokens"),
                                 reward_best_this_round=max(x["reward"] for x in tok if
                                                            (x["level"], x["run"], x["round"]) == k)))
        trunc = sum(1 for a in tok if a.get("finish") == "length")
        print(f"\n  {trunc} of {len(tok)} answers were truncated (finish=length)"
              + ("  <- the answer budget, not the model, is failing these" if trunc else ""))

    section("4. Calibration (does the agent know when it failed?)")
    if not results:
        print("  no verdict records in this log (it predates the instrumented agent.py)")
    else:
        wrong = [r for r in results
                 if (r["verdict"] == "VERIFIED") != (r["best"] >= FULL - 1e-9)]
        claimed = sum(r["verdict"] == "VERIFIED" for r in results)
        print(f"  {len(results)} level verdicts: {claimed} VERIFIED, {len(results) - claimed} "
              f"NOT VERIFIED; {len(wrong)} disagree with the checker"
              + ("  <- investigate" if wrong else "  (every claim matches the checker)"))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        for name, rows in (("rates", rate_rows), ("taxonomy", tax_rows), ("tokens", tok_rows)):
            if rows:
                fn = os.path.join(out_dir, f"{stem}_{name}.csv")
                keys = list(dict.fromkeys(k for r in rows for k in r))
                with open(fn, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)
                print(f"  wrote {fn}")
        if tok_rows:
            plot_tokens(tok_rows, os.path.join(out_dir, f"{stem}_tokens.png"))
    return rate_rows


def plot_tokens(rows, fn):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: tokens CSV written, no PNG)")
        return
    parts = ["instructions", "reference", "docs", "kernel", "feedback", "ledger"]
    colors = ["#8c8c8c", "#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"]
    runs = sorted({(r["level"], r["run"]) for r in rows})
    fig, axes = plt.subplots(1, len(runs), figsize=(max(4, 2.6 * len(runs)), 3.6), sharey=True,
                             squeeze=False)
    for ax, (lv, run) in zip(axes[0], runs):
        rs = [r for r in rows if r["level"] == lv and r["run"] == run]
        x = [r["round"] for r in rs]
        bottom = [0] * len(rs)
        for p, c in zip(parts, colors):
            v = [r[p] for r in rs]
            ax.bar(x, v, bottom=bottom, color=c, label=p, width=0.8)
            bottom = [b + y for b, y in zip(bottom, v)]
        ax.set_title(f"level {lv}, run {run}", fontsize=9)
        ax.set_xlabel("round")
        ax.set_xticks(x)
    axes[0][0].set_ylabel("input tokens")
    axes[0][-1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Where each round's input tokens went", fontsize=10)
    fig.tight_layout()
    fig.savefig(fn, dpi=140)
    print(f"  wrote {fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", help="directory for CSVs and the token chart")
    a = ap.parse_args()
    tables = {p: report(p, a.out) for p in a.logs}
    if len(tables) > 1:
        section("Side by side: solve rate and mean best reward per level")
        names = list(tables)
        levels = sorted({r["level"] for t in tables.values() for r in t})
        print("  level  " + "".join(os.path.basename(n)[:22].ljust(24) for n in names))
        for lv in levels:
            cells = []
            for n in names:
                r = next((x for x in tables[n] if x["level"] == lv), None)
                cells.append((f"{r['solved']}/{r['runs']}  mean {r['mean']:.2f}" if r else "-")
                             .ljust(24))
            print(f"  {lv:<6} " + "".join(cells))
        print("\n  Only compare logs made with the same --rounds/--samples/--context and model, and"
              "\n  only trust a difference on a zero-spread level, or one larger than the spread.")


if __name__ == "__main__":
    sys.exit(main())
