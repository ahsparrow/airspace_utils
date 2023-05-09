#!/usr/bin/env bash

echo "Generating overlays..."
python run.py overlay --max_alt 10400 ../airspace/airspace.yaml output/overlay_105.txt &
python run.py overlay --max_alt 19400 ../airspace/airspace.yaml output/overlay_195.txt &
python run.py overlay --max_alt 10400 ../airspace/airspace.yaml output/overlay_atzdz.txt --atzdz &

wait -n
echo "Done one"
wait -n
echo "Done two"
wait -n
echo "All done"
