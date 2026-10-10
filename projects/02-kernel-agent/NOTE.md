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

## How many runs, and why one per configuration is enough here

The README measured Qwen3 with sampling on and warns that one run is not a result. **On our seat the
server decodes greedily**, whatever `temperature` the request asks for. Measured:

* every round's 4 samples were byte-identical (8 of 8 rounds checked, on level 1; TODO: run the
  same check on levels 2–4 of `baseline.jsonl`);
* two independent runs of level 1 produced byte-identical kernels in all 8 rounds, the same errors in
  the same order, the same token counts per prompt part.

So the whole loop is deterministic: the same flags give the same trajectory, and a difference between
two configurations can only come from the change. We therefore ran **one run per configuration** with
`--samples 1`, and kept the 2-run baseline as the evidence for that choice.

A consequence worth stating on its own: the README reports level 2 solved 4 times in 5 **with
sampling**. Greedy, it is **0 of 1** — it settles on an out-of-bounds kernel and never leaves it.
Same model, same harness; the solve rate on level 2 came from sampling diversity, not from the loop.

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
| TODO | final, all helpful flags | 1–4 | | | |

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
