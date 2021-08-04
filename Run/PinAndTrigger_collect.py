import argparse
import GooseHDF5 as g5
import h5py
import numpy as np
import os
import shelephant
import sys
import tqdm

basename = os.path.splitext(os.path.basename(__file__))[0]

parser = argparse.ArgumentParser()
parser.add_argument("-A", "--min-A", type=int, help="Save events only with A > ...", default=200)
parser.add_argument("-o", "--output", type=str, help="Output file (appended)", default=basename + ".h5")
parser.add_argument("-e", "--error", type=str, help="Store list of corrupted files", default=basename + ".yaml")
parser.add_argument("files", type=str, nargs="*", help="Files to add")
args = parser.parse_args()
assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
assert len(args.files) > 0
args = parser.parse_args()

init = True
corrupted = []
existing = []

with h5py.File(args.output, "a") as output:

    for file in tqdm.tqdm(args.files):

        try:
            with h5py.File(file, "r") as data:
                pass
        except:
            corrupted += [file]
            continue

        with h5py.File(file, "r") as data:
            paths = list(g5.getdatasets(data))
            verify = g5.verify(data, paths)
            if paths != verify:
                corrupted += [file]
                continue


        with h5py.File(file, "r") as data:

            basename = os.path.basename(file)

            info = dict(
                stress = basename.split("stress=")[1].split("_")[0],
                A = basename.split("A=")[1].split("_")[0],
                id = basename.split("id=")[1].split("_")[0],
                incc = basename.split("incc=")[1].split("_")[0],
                element = basename.split("element=")[1].split(".hdf5")[0])

            # account for typo
            if "PushAndTrigger" in data["meta"]:
                meta = data["meta"]["PushAndTrigger"]
                root_meta = "/meta/PushAndTrigger"
            elif "PinAndTrigger" in data["meta"]:
                meta = data["meta"]["PinAndTrigger"]
                root_meta = "/meta/PinAndTrigger"
            else:
                raise IOError("Unknown input")

            if init:
                version = meta["version"].asstr()[...]
                version_dependencies = list(meta["version_dependencies"].asstr()[...])
                output["/meta/version"] = version
                output["/meta/version_dependencies"] = version_dependencies
                init = False
            else:
                assert version == meta["version"].asstr()[...]
                assert version_dependencies == list(meta["version_dependencies"].asstr()[...])

            assert int(info["incc"]) == meta["target_inc_system"][...]
            assert int(info["A"]) == meta["target_A"][...]
            assert int(info["element"]) == meta["target_element"][...]
            assert int(info["id"]) == int(os.path.splitext(str(meta["file"].asstr()[...]).split("id=")[1])[0])

            root = "/data/stress={stress:s}/A={A:s}/id={id:s}/incc={incc:s}/element={element:s}".format(**info)

            if root in output:
                existing += [file]
                continue

            A = meta["A"][...]

            source_datasets = [
                "{0:s}/file".format(root_meta),
                "{0:s}/target_stress".format(root_meta),
                "{0:s}/S".format(root_meta),
                "{0:s}/A".format(root_meta)]

            dest_datasets = ["/file", "/target_stress", "/S", "/A"]

            if A >= args.min_A:
                source_datasets = ["/disp/0", "/disp/1"] + source_datasets
                dest_datasets = ["/disp/0", "/disp/1"] + dest_datasets

            g5.copydatasets(data, output, source_datasets, dest_datasets, root)

shelephant.yaml.dump(args.error, dict(corrupted=corrupted, existing=existing), force=True)
