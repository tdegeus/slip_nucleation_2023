import argparse
import h5py
import numpy as np
import os
import sys
import GooseHDF5 as g5

# parser = argparse.ArgumentParser()
# parser.add_argument("-o", "--output", type=str, help="Output file (appended)", default="PinAndTrigger.h5")
# parser.add_argument("files", type=str, nargs="*", help="Files to add")
# args = parser.parse_args()


files = [
    "stress=0d6_A=800_id=939_incc=51_element=0.hdf5",
    "stress=1d6_A=800_id=169_incc=37_element=0.hdf5",
    "stress=1d6_A=800_id=224_incc=43_element=0.hdf5",
    # "stress=1d6_A=800_id=457_incc=41_element=0.hdf5",
    # "stress=1d6_A=800_id=481_incc=43_element=0.hdf5",
    # "stress=1d6_A=800_id=925_incc=47_element=0.hdf5",
    # "stress=1d6_A=800_id=978_incc=45_element=0.hdf5",
    "stress=2d6_A=800_id=027_incc=55_element=0.hdf5",
    # "stress=2d6_A=800_id=327_incc=47_element=0.hdf5",
    "stress=2d6_A=800_id=367_incc=51_element=0.hdf5",
    "stress=2d6_A=800_id=470_incc=41_element=0.hdf5",
    "stress=2d6_A=800_id=685_incc=47_element=0.hdf5",
    # "stress=3d6_A=800_id=013_incc=43_element=0.hdf5",
    "stress=3d6_A=800_id=155_incc=45_element=0.hdf5",
    "stress=3d6_A=800_id=286_incc=45_element=0.hdf5",
    "stress=3d6_A=800_id=354_incc=37_element=0.hdf5",
    "stress=3d6_A=800_id=380_incc=39_element=0.hdf5",
    # "stress=3d6_A=800_id=431_incc=47_element=0.hdf5",
    "stress=3d6_A=800_id=522_incc=43_element=0.hdf5",
    "stress=3d6_A=800_id=658_incc=43_element=0.hdf5",
    "stress=3d6_A=800_id=899_incc=39_element=0.hdf5",
    # "stress=4d6_A=800_id=106_incc=29_element=0.hdf5",
    # "stress=4d6_A=800_id=197_incc=47_element=0.hdf5",
    "stress=4d6_A=800_id=302_incc=41_element=0.hdf5",
    "stress=4d6_A=800_id=546_incc=55_element=0.hdf5",
    # "stress=4d6_A=800_id=673_incc=39_element=0.hdf5",
    # "stress=4d6_A=800_id=780_incc=47_element=0.hdf5",
    "stress=4d6_A=800_id=870_incc=51_element=0.hdf5",
    # "stress=5d6_A=800_id=117_incc=27_element=0.hdf5",
    "stress=5d6_A=800_id=444_incc=35_element=0.hdf5",
    # "stress=5d6_A=800_id=533_incc=39_element=0.hdf5",
    # "stress=5d6_A=800_id=586_incc=53_element=0.hdf5",
    # "stress=5d6_A=800_id=755_incc=33_element=0.hdf5",
    # "stress=5d6_A=800_id=768_incc=41_element=0.hdf5",
    # "stress=5d6_A=800_id=792_incc=37_element=0.hdf5",
    # "stress=5d6_A=800_id=817_incc=43_element=0.hdf5",
    # "stress=5d6_A=800_id=831_incc=35_element=0.hdf5",
    # "stress=6d6_A=800_id=066_incc=37_element=0.hdf5",
    # "stress=6d6_A=800_id=078_incc=27_element=0.hdf5",
    # "stress=6d6_A=800_id=314_incc=43_element=0.hdf5",
    # "stress=6d6_A=800_id=642_incc=43_element=0.hdf5",
    # "stress=6d6_A=800_id=844_incc=43_element=0.hdf5",
]

init = True

with h5py.File("PinAndTrigger.h5", "a") as output:

    for file in files:

        with h5py.File(file, "r") as data:

            info = dict(
                stress = file.split("stress=")[1].split("_")[0],
                A = file.split("A=")[1].split("_")[0],
                id = file.split("id=")[1].split("_")[0],
                incc = file.split("incc=")[1].split("_")[0],
                element = file.split("element=")[1].split(".hdf5")[0])

            if init:
                version = data["/meta/PushAndTrigger/version"].asstr()[...]
            else:
                assert version == data["/meta/PushAndTrigger/version"].asstr()[...]

            # todo: save version and dependencies

            root = "/stress={stress:s}/A={A:s}/id={id:s}/incc={incc:s}/element={element:s}".format(**info)

            if root in output:
                print('Skipping', root)
                continue

            g5.copydatasets(data, output, ["/disp/0", "/disp/1", "/meta/PushAndTrigger/file"], ["/disp/0", "/disp/1", "/file"], root=root)

            # todo add metadata?

