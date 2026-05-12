#!/usr/bin/env bash

shopt -s nullglob

for f in snap_*.txt; do
    i=${f#snap_}
    i=${i%.txt}

    correct="snap_correct_${i}.txt"

    if [[ -f "$correct" ]]; then
        if ! diff -q "$f" "$correct" >/dev/null; then
            echo "DIFF FOUND for snapshot $i"
        fi
    fi
done

if [[ -f stats.txt && -f stats_correct.txt ]]; then
    if ! diff -q stats.txt stats_correct.txt >/dev/null; then
        echo "DIFF FOUND for stats.txt"
    fi
fi
