## YAIXM

Utilities for processing YAIXM airspace data (see
[GitHub](https://github.com/ahsparrow/airspace).

### Utilities

Use the run.py script for various utilities, `./run.py -h` to get a list

To validate a YAIXM file against the schema:

    ./run.py check airspace.yaml

To generate a ASSelect airspace files

    ./run.py release [--note RELEASE_NOTE] yaixm_dir yaixm.json openair.txt

To generate the ASSelect overlay file (takes a few minutes)

    ./run.py overlay ../airspace/airspace.yaml ~/src/asselect/src/assets/overlay.txt
