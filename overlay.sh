#!/usr/bin/env bash

echo "Generating overlays..."
uv run cli overlay --max_alt 10400 ../airspace/airspace.yaml output/overlay_105.txt &
uv run cli overlay --max_alt 19400 ../airspace/airspace.yaml output/overlay_195.txt &
uv run cli overlay --max_alt 10400 ../airspace/airspace.yaml output/overlay_atzdz.txt --atzdz &

wait -n
echo "Done one"
wait -n
echo "Done two"
wait -n
echo "All done"
