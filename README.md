## YAIXM

Utilities for processing YAIXM airspace data (see
[GitHub](https://github.com/ahsparrow/airspace)).

### Utilities

Use the run.py script for various utilities, `uv run cli -h` to get a list

To validate a YAIXM file against the schema:

    uv run cli check airspace.yaml

To generate a ASSelect airspace files

    uv run cli release [--note RELEASE_NOTE] yaixm_dir output/yaixm.json output/openair.txt

To generate the ASSelect overlay files (takes a few minutes)

    uv run cli overlay --max_alt 10400 ../airspace/airspace.yaml output/overlay_105.txt
    uv run cli overlay --max_alt 19400 ../airspace/airspace.yaml output/overlay_195.txt
    uv run cli overlay --max_alt 10400 --atzdz ../airspace/airspace.yaml output/overlay_atzdz.txt

To deploy new airspace files copy files from output directory to
`../asselect3/data` and follow instructions to deploy asselect3.
