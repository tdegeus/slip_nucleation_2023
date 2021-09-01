import sys, os, re, subprocess, shutil, h5py

import numpy as np

# --------------------------------------------------------------------------------------------------


def check(data):

    if "completed" not in data["meta"]:
        print(data.filename, 'Error: "/meta/completed" not found.')
        return

    if "sync-t" not in data:
        print(data.filename, 'Error: "/sync-t/..." not found.')
        return

    inc = data["/sync-A/stored"][...]

    idx0 = data[f"/sync-A/plastic/{np.min(inc):d}/idx"][...]
    idx = data[f"/sync-A/plastic/{np.max(inc):d}/idx"][...]

    if np.sum(idx0 != idx) != len(idx):
        print(data.filename, 'Error: "/sync-A/..." not system spanning.')
        return

    inc = data["/sync-t/stored"][...]

    idx = data[f"/sync-t/plastic/{np.max(inc):d}/idx"][...]

    if np.sum(idx0 != idx) != len(idx):
        print(data.filename, 'Error: "/sync-t/..." not system spanning.')
        return

    print(data.filename)


# --------------------------------------------------------------------------------------------------

files = sorted(
    list(
        filter(
            None,
            subprocess.check_output("find . -iname '*id*.hdf5'", shell=True)
            .decode("utf-8")
            .split("\n"),
        )
    )
)

# --------------------------------------------------------------------------------------------------

for file in files:

    data = h5py.File(file, "r")

    check(data)

    data.close()
