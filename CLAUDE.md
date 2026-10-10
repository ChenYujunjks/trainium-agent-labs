# Handoff: Bruce's Hack the Chip run (project 2, the kernel agent)

You are picking up a hackathon run already in progress. Read this whole file before acting.
The user is Bruce (NYU MSCS). He works in Chinese; answer him in Chinese. He is working solo and
wants you to drive the terminal so he does not have to copy commands back and forth.

## Where things run

* **This Mac** holds a clone of the team fork, `ChenYujunjks/trainium-agent-labs`, branch
  `bruce/instrumentation`. Edit code here, commit, push.
* **The pod `seat-88`** (AWS EKS cluster `hack-hyd`, region `ap-south-2`) has one Trainium chip, the
  model server, and the NKI simulator. All runs happen there. Reach it only with kubectl:

  ```bash
  kubectl exec seat-88 -- bash -lc 'cd /workspace/projects/02-kernel-agent && ./status.sh'
  ```

  The pod's repo is `/workspace`, with remote `team` pointing at the fork. To deploy code:
  push from the Mac, then `kubectl exec seat-88 -- bash -lc 'cd /workspace && git pull team bruce/instrumentation'`.
  The pod never pushes; it holds no GitHub token.
* **AWS credentials** are temporary env vars Bruce pasted into this terminal. Never write them to a
  file, never echo them, never commit them (GitHub scans pushes; the repo is public). If kubectl says
  `ExpiredToken`, stop and ask Bruce to paste the newest block from the workshop channel.
* The model server (Qwen3-8B, vLLM-Neuron, context 8192) is already running inside the pod at
  `http://localhost:8000/v1`. If `curl -s localhost:8000/v1/models` fails inside the pod, start it with
  `cd /workspace && ./serve.sh` (about 4 minutes). The pod env sets `KERNEL_AGENT_BASE_URL` and
  `KERNEL_AGENT_MODEL`; if a non-interactive shell lacks them, pass
  `--base http://localhost:8000/v1 --model Qwen/Qwen3-8B` to agent.py.

## Rules that cost us time to learn

1. **One agent.py at a time.** All runs share one model server. Check `pgrep -af agent.py` before
   starting anything. Bruce once ended up with three copies running by accident.
2. **Every run goes in the background, to its own log**, because a dropped kubectl session kills
   foreground jobs:
   `nohup python agent.py ARGS --log NAME.jsonl > NAME.log 2>&1 < /dev/null &`
3. **The server is deterministic (greedy).** Measured: every round's 4 samples were byte-identical,
   and two independent runs of level 1 produced byte-identical kernels in all 8 rounds. So:
   use `--samples 1` (4 samples = 4 copies), and **one run per configuration is a valid comparison**.
   Never use `--repeat` here; it only re-plays the same run.
4. **Round time is ~50 s, all of it generation** (~0.18 s/token, ~290 tokens per answer). A level is
   up to 8 rounds, so about 7 minutes; all 4 levels about 25 minutes.
5. **Change one thing per experiment, as an opt-in flag.** Defaults must stay byte-identical to the
   baseline prompt. `/tmp`-style check used so far: 49 prompt variants compared against the original
   `first_prompt`/`repair_prompt` — keep that property when you edit prompts.
6. Logs and reports live in the pod's `/workspace/projects/02-kernel-agent/`. To bring one to the Mac:
   `kubectl cp seat-88:/workspace/projects/02-kernel-agent/FILE ./runs/bruce/FILE`

## What the code on this branch adds (all in projects/02-kernel-agent/)

* `agent.py`
  * every prompt is a list of named parts (instructions, reference, docs, kernel, feedback, ledger);
    each attempt logs exact per-part input tokens (server `/tokenize`), server usage, finish reason,
    seconds; each level ends with a `type: result` record and a `VERIFIED` / `NOT VERIFIED` verdict
    that states it is CPU-simulator only
  * experiment flags, all OFF by default:
    `--api-card level` drop nc_matmul material from the card on levels 1-2;
    `--ap-example` a generic strided-view + nl.sum example on levels 1-2;
    `--repair-card` short list of real calls in a repair prompt after an invented-name failure;
    `--shape-hint` on a dma_copy size mismatch, name the input tensor and give it its own tile
  * `enrich()` additions (unconditional; neither error occurs in the baseline): missing-argument
    TypeErrors (e.g. tensor_scalar) and PSUM->HBM dma_copy
* `analyze.py LOG [LOG ...] [--out DIR]` — solve rate, failure taxonomy, per-round token table,
  calibration; reads old and new log formats. Taxonomy counts are per ATTEMPT, so the baseline (4
  identical samples per round) is inflated 4x — divide by 4 to get rounds.
* `status.sh [LOG]` — one-shot progress report for a run.

## Results so far (Qwen3-8B on seat-88, rounds 8, context 8192)

| run | level | flags | best | failure it ended on / note |
|---|---|---|---|---|
| baseline (`baseline.jsonl`, samples 4) | 1 | none | 0.30 | invented_api; wrote nc_matmul for average pooling |
| baseline | 2 | none | 0.30 | out_of_bounds. README reports 4/5 solved WITH sampling; greedy = 0 |
| baseline | 3 | none | 0.30 | dma_copy size mismatch x4, stopped |
| baseline | 4 | none | 0.62 | tile_too_big (1 of 4 shapes) |
| exp1 (`exp1_card.jsonl`) | 1 | `--api-card level` | 0.30 | invented_api 5 rounds -> 0; wrong_buffer gone; now out_of_bounds x4 |
| exp2 (`exp2_repaircard.jsonl`) | 1 | `--repair-card` | 0.30 | invented_api 5 -> 2; reached nisa.tensor_scalar but missing args (enrich now covers that) |
| exp3 (`exp3_shapehint.jsonl`) | 3 | `--shape-hint` | 0.30 | dma mismatch 4 -> 1; now `cannot reshape` x4 (README's documented level-3 wall) |

Other measured facts worth the write-up:
* The largest prompt in the whole baseline was 716 tokens, under 9% of the 8192 window; 0 of 124
  answers truncated. The context budget is not the constraint here; content is. Repair prompts carry
  no docs at all.
* Pattern in every experiment: the score stays at 0.30 but the failure MOVES to the next wall. Report
  the failure transitions, not just the score.

## Next steps, in order (it was 14:30 when this was written)

1. Confirm the pod has the latest commit: `git -C /workspace log --oneline -1`.
2. Level 3 now stalls on `cannot reshape array`. Print the last attempt's code and feedback from
   `exp3_shapehint.jsonl`, and improve the message for that error (in `enrich()`, or a new flag) so
   it names the exact slicing change. Re-run level 3 with `--shape-hint` plus the fix.
3. Level 1 under `--api-card level` stalls on out_of_bounds; do the same for that message.
4. Around 15:15: one full run of all four levels with every flag that helped, `--samples 1`,
   log `final.jsonl`. Compare with `python analyze.py baseline.jsonl final.jsonl --out report`.
5. Write the one-page note (NOTE.md): what ran, on what, how many runs (and why one suffices:
   determinism, with the evidence), the failure transitions table, the token budget finding, the
   calibration result (every verdict matched the checker), and honest limits: simulator only, never
   timed on the device; `--api-card level` both removes matmul text and shortens the prompt, so the
   two causes are confounded.
6. Hand-in, per README Part 4: the checker and its reasoning, the attempt logs, the one-page note.
   Copy logs to the Mac with `kubectl cp` into `runs/bruce/`, commit, push.
