#!/usr/bin/env bash
# Keep the computer awake (no suspend, hibernate, lid-close sleep or idle sleep)
# while the w5 pipeline is running; releases the lock by itself when it ends.
# Usage:  setsid workshops/w5/keep_awake.sh > /dev/null 2>&1 &
# Check:  systemd-inhibit --list      Stop early:  pkill -f keep_awake.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$HERE/artifacts/experiments/run_all.pid"
PID="$(cat "$PID_FILE")"
exec systemd-inhibit \
  --what=sleep:idle:shutdown:handle-lid-switch:handle-suspend-key:handle-hibernate-key \
  --who="w5 GAN experiments" --why="Training GANs (pipeline pid $PID)" --mode=block \
  bash -c "while kill -0 $PID 2>/dev/null; do sleep 60; done"
