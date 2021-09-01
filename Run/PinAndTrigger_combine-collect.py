import argparse
import os
import shutil

import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm

basename = os.path.splitext(os.path.basename(__file__))[0]

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o", "--output", type=str, help="Output file (appended)", default=basename + ".h5"
)
parser.add_argument("files", type=str, nargs="*", help="Files to add")
args = parser.parse_args()
assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
assert len(args.files) > 0
args = parser.parse_args()

shutil.copyfile(args.files[0], args.output)

with h5py.File(args.output, "a") as output:

    for file in tqdm.tqdm(args.files[1:]):

        with h5py.File(file, "r") as data:

            for key in ["/meta/version", "/meta/version_dependencies"]:
                assert g5.equal(output, data, key)

            paths = list(g5.getdatasets(data))
            paths.remove("/meta/version")
            paths.remove("/meta/version_dependencies")

            g5.copydatasets(data, output, paths)
