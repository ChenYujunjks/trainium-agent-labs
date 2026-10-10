#!/usr/bin/env bash
# One command, everything needed to judge a run: is it alive, where is it, what has it concluded.
#
#   ./status.sh                 the newest *.log in this folder
#   ./status.sh baseline.log    a specific run log
cd "$(dirname "$0")"
LOG=${1:-$(ls -t *.log 2>/dev/null | head -1)}
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then echo "no run log found here"; exit 1; fi

echo "=== processes ==="
pgrep -af "agent.py" || echo "(no agent.py running -- the run has finished or was never started)"

echo; echo "=== $LOG: progress and verdicts ==="
grep -E "^#+ run|^=+ level|VERDICT|STOPPING|SOLVED on|not solved in" "$LOG" | tail -30

echo; echo "=== $LOG: last 4 lines ==="
tail -4 "$LOG" | cut -c1-300

if grep -q "over [0-9]* runs\|=== summary ===" "$LOG"; then
  echo; echo "=== $LOG: summary ==="
  sed -n '/=== summary ===/,$p' "$LOG" | tail -20
fi
