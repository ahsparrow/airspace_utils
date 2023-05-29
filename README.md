## YAIXM

Utilities for processing YAIXM airspace data (see
[GitHub](https://github.com/ahsparrow/airspace).

### Utilities

Use the run.py script for various utilities, `./run.py -h` to get a list

To validate a YAIXM file against the schema:

    ./run.py check airspace.yaml

To generate a ASSelect airspace files

    ./run.py release [--note RELEASE_NOTE] yaixm_dir output/yaixm.json output/openair.txt

To generate the ASSelect overlay files (takes a few minutes)

    ./run.py overlay --max_alt 10400 ../airspace/airspace.yaml output/overlay_105.txt
    ./run.py overlay --max_alt 19400 ../airspace/airspace.yaml output/overlay_195.txt
    ./run.py overlay --max_alt 10400 --atzdz ../airspace/airspace.yaml output/overlay_atzdz.txt

To deploy new airspace files to asselect

    ./run deploy output
