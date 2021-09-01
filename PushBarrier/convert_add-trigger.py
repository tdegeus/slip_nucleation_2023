import subprocess

import h5py

files = sorted(
    list(
        filter(
            None,
            subprocess.check_output("find . -iname 'id*.hdf5'", shell=True)
            .decode("utf-8")
            .split("\n"),
        )
    )
)

N = int(3 ** 6 * 2)
i = int(3 ** 6)

for file in files:

    print(file)

    with h5py.File(file, "a") as data:
        data["/trigger/i"] = i
        data["/completed"] = 0
