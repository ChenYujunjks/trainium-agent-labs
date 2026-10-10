# The kernel agent on a greedy server: what we ran, what moved, what did not

*Bruce, NYU. Hack the Chip, 10 Oct 2026. Project 2. DRAFT — final-run numbers still to fill in (marked TODO).*

## What ran, on what

* **Model:** Qwen3-8B, served by vLLM-Neuron on one Trainium2 chip (seat pod `seat-88`, TP 2, context
  8192), thinking off.
* **Checker:** layer 1 of `nkibench.py` only — static rules, then `nki.simulate` on the CPU against a
  NumPy reference on 1–4 shapes per level. **No kernel was run or timed on the device.** Every
  `VERIFIED` below means "correct in the CPU simulator", and the agent's own verdict line says so.
* **Agent:** the repo's `agent.py` loop (8 rounds, repair the latest attempt, failure ledger, stop
  after 4 identical failures), plus our instrumentation and four opt-in prompt experiments.
* **Levels:** 1 average pooling, 2 transpose, 3 single-tile matmul, 4 tiled matmul.

## How many runs, and what "deterministic" turned out to mean

The README measured Qwen3 with sampling on and warns that one run is not a result. **On our seat the
server decodes greedily**, whatever `temperature` the request asks for. Measured:

* every round's 4 samples were byte-identical (8 of 8 rounds checked, on level 1; TODO: run the
  same check on levels 2–4 of `baseline.jsonl`);
* two independent level-1 runs produced byte-identical kernels in all 8 rounds.

**We first concluded that one run per configuration was enough. That was wrong.** Level 3's round-0
prompt is the same 2477 characters in every run, yet across 8 runs it produced **4 different answers**
(fingerprints `233796be`, `a0404a32`, `3be04b6f`, `1c024b99`). Within one of those starting points the
trajectory is repeatable: the three fingerprints that recurred (`3be04b6f`, `a0404a32`, `1c024b99`)
each reproduced the same failures in the same order until the feedback text itself changed. So the
loop is deterministic *given its first answer*, and the first answer varies from run to run. We do not
know why (a server restart or numerical non-determinism are both candidates; not investigated).

Consequences: a single run per configuration cannot be compared with another unless their round 0
matches. Where it does match, the comparison is clean (experiment 5 below). Otherwise we repeat: the
final level-3 configuration was run **3 times**.

The README reports level 2 solved 4 times in 5 **with sampling**. Our one greedy level-2 run is
**0 of 1**, ending on an out-of-bounds kernel. We have not repeated it, so this shows only that the
README's rate does not transfer to a single greedy run, not what the greedy rate is.

Cost: ~50 s per round, essentially all of it generation (~0.18 s per output token, ~290 tokens per
answer). The CPU simulation is negligible next to it.

## The context budget was not the constraint

The challenge frames the 8192-token window as the hard part. We logged every prompt part with the
server's own tokenizer:

* the largest prompt in the baseline was **716 tokens, under 9% of the window**;
* **0 of 124** answers were truncated;
* first prompts are dominated by docs (514 of ~680 tokens); **repair prompts carry no docs at all** —
  only the previous kernel (~260–370), the checker's message (~45–190), and later the ledger.

So the question on this setup is not what to cut, it is what to put in. Repair rounds run with over
90% of the window empty and no reference for the API the model keeps getting wrong.

## Baseline: where each level stalls

| level | best | stalls on | what the model actually did |
|---|---|---|---|
| 1 pooling | 0.30 | invented API names | wrote `nc_matmul` for average pooling; then `nisa.multiply`, which does not exist |
| 2 transpose | 0.30 | out-of-bounds index | the same out-of-bounds access 4 rounds running, then stopped (TODO: quote the kernel) |
| 3 matmul, 1 tile | 0.30 | dma_copy size mismatch | one (128, 512) SBUF tile reused for lhsT (128, 64), rhs **and** the output |
| 4 matmul, tiled | 0.62 | tile over 128 rows | a tile taller than 128 partitions; passes 1 of 4 shapes |

Failure modes per round (baseline run 1): level 1 — invented API 5, size mismatch 2, wrong buffer 1;
levels 2–4 — the single mode above, repeated until the stop rule fired.

## Experiments: one change each, opt-in flags, default prompts unchanged

Each flag leaves the default prompt byte-identical to the baseline (checked on 49 prompt variants), so
the baseline is reproducible and each run differs from it in exactly one way.

| exp | flag | level | hypothesis | score | how the failure moved |
|---|---|---|---|---|---|
| 1 | `--api-card level` | 1 | the card's long `nc_matmul` paragraph tells the model a pooling kernel is a matmul | 0.30 → 0.30 | invented API **5 → 0**, wrong buffer **1 → 0**; now out-of-bounds ×4 |
| 2 | `--repair-card` | 1 | repair prompts have no API list, so the model guesses names | 0.30 → 0.30 | invented API **5 → 2**; reached the right call, `nisa.tensor_scalar`, with arguments missing |
| 3 | `--shape-hint` | 3 | the size-mismatch message names numbers, not the tensor, and its generic "128×512 piece" example points away from the fix | 0.30 → 0.30 | size mismatch **4 → 1**; now `cannot reshape` ×4 — the level-3 wall the README documents |
| 4 | `--reshape-hint` | 3 | "Do not reshape" blames code the model never wrote: the error is raised inside `nc_matmul` because the PSUM destination was `(1, 64)` | 0.30 → 0.30 | `cannot reshape` **gone**. Given only the numbers `(64, 512)` the model wrote `rhs.shape[0]` and got `(64, 128)`; given the expression `(lhsT.shape[1], rhs.shape[1])` it got the shape right. Next wall: the result copied into the (128, 512) input tile, `32768 could not be broadcast to 65536` |
| 5 | `--copy-hint` | 3 | the broadcast message talks about slices; the real fault is an input tile reused for the result | **0.30 → 1.00 in 2 of 3 runs** | see below |
| TODO | final, all helpful flags | 1–4 | | | |

**Experiment 5, level 3, three runs with `--shape-hint --reshape-hint --copy-hint`:**

| run | round-0 fingerprint | result | path |
|---|---|---|---|
| rep1 | `3be04b6f` | **VERIFIED 1.00, round 4** | dma size mismatch → `cannot reshape (128,512)` → broadcast 65536 → correct |
| rep2 | `a0404a32` | **VERIFIED 1.00, round 5** | 1-D tile ×2 → `cannot reshape (1,64)` → broadcast 65536 → correct |
| rep3 | `1c024b99` | 0.30, stopped as cycling | 1-D tile error 4 rounds running; never reached the new hints |

The cleanest pair: rep1 and the earlier experiment 5 run share the fingerprint `3be04b6f` and the same
first three failures; the only difference was `--copy-hint`, and the run without it stalled on the
broadcast error while the run with it was correct one round later. rep3's starting point also stuck
in an earlier run (experiment 6 log), so that failure repeats rather than being noise. **Limits:** three
runs, two distinct successful starting points, no hint-free control from the same starting point for
rep2/rep3, simulator only. The solved kernel is reported as MEMORY BOUND at 19.7 Flops/Byte against a
ridge of 222, so it is correct but far from efficient.

**Experiment 6, `--line-hint` added (appends the failing source line), level 3, three more runs:** all
three VERIFIED 1.00, each in 4 rounds (fingerprints `a0404a32` ×2, `3be04b6f`). **This did not test
the flag on the wall it was built for:** none of the three started from `1c024b99`, the start that
stalled on the 1-D tile error in rep3 and experiment 6's earlier log. The two `a0404a32` runs left the
1-D wall after one round, where the earlier `a0404a32` run (rep2) needed two; that is two runs against
one and is not evidence. What stands: level 3 solved in 5 of 6 runs with all hints on, the one failure
being the start the line hint was meant for. TODO: run until `1c024b99` recurs, with and without
`--line-hint`.

**Experiment 7, level 1, `--api-card level` with and without `--line-hint`, three runs each (alternating):**

| run | flags | round-0 fingerprint | rounds | ended on |
|---|---|---|---|---|
| l1ctl1 | card | `cf007a29` | 5, cycling | out-of-bounds (index up to 127 on a 32-long axis) ×4 |
| l1ctl2 | card | `cf007a29` | 5, cycling | out-of-bounds ×4 |
| l1ctl3 | card | `739f47f0` | 5, cycling | out-of-bounds ×4 |
| l1line1 | card + line | `739f47f0` | 6, cycling | dma_copy size mismatch ×4 (out-of-bounds once) |
| l1line2 | card + line | `cf007a29` | 8, out of rounds | dma_copy mismatch, then `Tile and source shape mismatch` |
| l1line3 | card + line | `cf007a29` | 8, cycling | dma_copy mismatch, then `Tile and source must have the same number of elements` |

All six scored 0.30; none solved. **But the out-of-bounds wall disappeared with the line hint:** the three
controls stopped on it four rounds running, and none of the three line-hint runs did, including the
same-fingerprint pairs (`cf007a29`: 2 controls vs 2 line-hint; `739f47f0`: 1 vs 1). In every run the
line-hint model saw the out-of-bounds error at most once and then changed its code. The failure moved on
to a dma_copy size mismatch — the same pattern as every earlier experiment. The error text of the new
failures points at the line (`The error is raised by line 18 of your kernel: ...`). Limits: three
runs per arm, simulator only, and only the fingerprint pairs are like-for-like.

**Experiment 8, level 1, `--shape-hint` added to experiment 7's flags (card + line), three runs:** all
three started from `cf007a29` and were identical round for round (a case where "deterministic given
the first answer" held exactly). All scored 0.30, stopped as cycling after 6 rounds. **The dma_copy size
mismatch was gone after round 0 in all three**; the two same-fingerprint runs without `--shape-hint`
(`l1line2`, `l1line3`) kept returning to it for rounds 2–5. The new wall is the same in all three:
the model invents `nl.reshape` (round 1), then calls `nl.transpose(tile, (1, 0))`, which the
simulator rejects as `Unknown dtype: (1, 0)`, and repeats that call for four rounds. It is reaching
for reshape/transpose to carve each pooling window out of a tile, i.e. it lacks the strided
access-pattern idiom; this is the capability wall the README describes for level 1. Next candidate:
the existing `--ap-example`.

**Experiment 9, level 1, `--ap-example` added to experiment 8's flags, three runs:** no run solved
(0.30 ×3, all stopped as cycling). The example changes the round-0 prompt, so the three starting
points (`f623f401`, `12d65785`, `c93df708`) match none of experiment 8's (`cf007a29`) and the arms
are **not like-for-like**. What the logs do show: the model began calling `ap()`, which it had not
used before, so it read the example; two runs then repeated `ap() pattern has invalid partition
stride. Partition step 1 must equal tensor free dimension size` for six rounds, and one repeated the
1-D tile error for four. There is no evidence that `--ap-example` helps, so it is left out of the final
configuration. The failure moved to a harder idiom, which is a transition, not progress in score.

**The score did not move once, and the agent got further every time.** Each change removed the
failure it targeted and exposed the next one. Reporting only the reward would say "nothing worked";
the failure transitions say each fix worked and the next wall is now visible. This is the README's
"the failure moves rather than vanishes", measured three times.

Two messages were added to `enrich()` directly rather than behind a flag, because experiment 2 is the
first place their errors ever occurred (the baseline log contains neither): a missing-argument
`TypeError` now gets the call's real signature and a worked `tensor_scalar` example, and a PSUM→HBM
`dma_copy` gets the `tensor_copy` step it skipped. On a deterministic server the baseline trajectory
cannot reach them, so the baseline comparison stands.

## The checker, and what it accepts and rejects

Unchanged from the repo (`nkibench.py`): a kernel scores 0.1 for parsing, 0.2 for passing the static
rules (no framework call that does the whole job, partition ≤ 128, `@nki.jit`, the right entry name),
0.2 for running in the simulator, and 0.5 prorated by shapes matching the NumPy reference within 2%
of the output's RMS. Inputs are snapshotted so a kernel that writes into its input fails. What we
changed is not what the checker accepts but **what it says** when it rejects: the message is the
instruction the model acts on, and three of our four changes are message changes.

## Calibration

Every level ends with a verdict derived only from the checker. Across all runs: TODO verdicts, 0
`VERIFIED` on a kernel that failed, 0 disagreements with the checker. The agent never claimed a pass
it did not have.

## Honest limits

* Simulator only. No latency, no profile, nothing measured on the device.
* `--api-card level` both removes matmul text **and** shortens the prompt by 27%; the README found
  shorter prompts help on their own. We cannot separate the two causes from one run.
* One model, one seat. Greedy decoding is a property of this server's configuration, not of Qwen3;
  with sampling on, the variance the README measured returns and single runs stop being enough.
* Levels 5–8 were not attempted.

## Reproduce

```bash
git clone https://github.com/ChenYujunjks/trainium-agent-labs && cd trainium-agent-labs
git checkout Project2 && cd projects/02-kernel-agent
python agent.py --all --rounds 8 --samples 1 --context 8192 --log baseline.jsonl     # baseline
python agent.py --level 1 --rounds 8 --samples 1 --context 8192 --api-card level --log exp1.jsonl
python analyze.py baseline.jsonl exp1.jsonl --out report
```

Attempt logs: `runs/bruce/*.jsonl` (TODO: copy from the pod).
