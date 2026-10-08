#!/bin/sh
# Full pipeline for one model:  sh chain.sh <tag> [resume-step]   (settings per tag in env.sh)
# band probe -> band -> pain axis -> prep -> E1 -> E3 -> E4 -> E2 (steering, embedded) -> video traces.
cd "$(dirname "$0")"
P=../.venv/bin/python
. ./env.sh "$1" || exit 1
T=out/$1
FROM=${2:-}      # optional: resume at this step name (earlier steps are skipped)
run() {
  name=$1; shift
  if [ -n "$FROM" ]; then
    if [ "$name" = "$FROM" ]; then FROM=; else echo "$(date -Is) skip $name"; return 0; fi
  fi
  echo "$(date -Is) $name"
  "$@" > ${T}_$name.log 2>&1; rc=$?
  if [ $rc -ge 128 ]; then      # killed by a signal (a segfault is 139): retry once, keeping the first log
    mv ${T}_$name.log ${T}_$name.crash$rc.log
    echo "$(date -Is) retry $name after signal exit $rc"
    "$@" > ${T}_$name.log 2>&1; rc=$?
  fi
  [ $rc -eq 0 ] || { echo "$(date -Is) FAILED $name (exit $rc)"; exit 1; }
}
run smoke $P smoke.py
run band $P band_probe.py
run bandjson $P band_json.py ${T}_band_stats.npz
run p1 $P p1_axis.py
run s0 $P s0_prep.py
run s1 $P s1_map.py
run s3 $P s3_split.py
run s4 $P s4_persist.py
run s2c $P s2c_steer.py
run s2b $P s2b_embedded.py
run vt $P v_traces.py
run vcap $P v_capacity.py
run vlay $P v_layers.py
echo done > ${T}_chain.done
echo "$(date -Is) chain done"
