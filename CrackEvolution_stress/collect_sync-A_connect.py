r"""
For avalanches synchronised at avalanche area `A`,
compute the distance between a new yielding event and the largest connected block.

Usage:
    collect_sync-A_connect.py [options] <files>...

Arguments:
    <files>     Files from which to collect data.

Options:
    -o, --output=<N>    Output file. [default: output.hdf5]
    -i, --info=<N>      Path to EnsembleInfo. [default: EnsembleInfo.hdf5]
    -f, --force         Overwrite existing output-file.
    -h, --help          Print help.
"""
import os
import sys

import click
import docopt
import GooseEYE as eye
import h5py
import numpy as np

# ==================================================================================================
# get files
# ==================================================================================================

args = docopt.docopt(__doc__)

files = args["<files>"]
output = args["--output"]

for file in files:
    if not os.path.isfile(file):
        raise OSError(f'"{file:s}" does not exist')

if not args["--force"]:
    if os.path.isfile(output):
        print(f'"{output:s}" exists')
        if not click.confirm("Proceed?"):
            sys.exit(1)

# ==================================================================================================
# get constants
# ==================================================================================================

with h5py.File(files[0], "r") as data:
    plastic = data["/meta/plastic"][...]
    nx = len(plastic)

if nx % 2 == 0:
    mid = nx / 2
else:
    mid = (nx - 1) / 2

# ==================================================================================================
# support functions
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# get the renumber index
# --------------------------------------------------------------------------------------------------


def getRenumIndex(old, new, N):

    idx = np.tile(np.arange(N), (3))
    ii = old + N - new
    jj = old + 2 * N - new
    return idx[ii:jj]


# --------------------------------------------------------------------------------------------------
# compute distance of all newly yielded cells, to the biggest cluster
# --------------------------------------------------------------------------------------------------


def compute_distance(cracked, active, clusters):

    nx = active.size

    if nx % 2 == 0:
        mid = int(nx / 2)
    else:
        mid = int((nx - 1) / 2)

    sizes = clusters.sizes()
    centers = clusters.centers()

    length = np.argmax(sizes[1:]) + 1
    center = int(np.argwhere(centers == length).ravel())
    size = sizes[length]

    renum = getRenumIndex(center, mid, nx)

    if size % 2 == 0:
        dl = int(size / 2 - 1)
        dr = int(size / 2)
    else:
        dl = int((size - 1) / 2)
        dr = int((size - 1) / 2)

    iactive = np.argwhere(active[renum]).ravel()

    d = nx * np.ones((2, nx), dtype="int")
    d[0, iactive] = mid - iactive - dl
    d[1, iactive] = iactive - mid - dr
    d = np.where(d >= 0, d, nx)
    d = np.min(d, axis=0)

    d0 = np.empty((nx), dtype="int")
    d0[renum] = d

    return d0[np.argwhere(active).ravel()]


# --------------------------------------------------------------------------------------------------
# check "compute_distance"
# --------------------------------------------------------------------------------------------------

if True:

    c = np.array([1, 1, 0, 0, 0, 1])
    a = np.array([0, 0, 1, 1, 1, 0])
    assert np.all(compute_distance(c, a, eye.Clusters(c, periodic=True)) == np.array([1, 2, 1]))

    c = np.array([1, 1, 0, 0, 0, 1, 1])
    a = np.array([0, 0, 1, 1, 1, 0, 0])
    assert np.all(compute_distance(c, a, eye.Clusters(c, periodic=True)) == np.array([1, 2, 1]))

    c = np.array([1, 1, 0, 0, 0, 1, 1])
    a = np.array([0, 0, 1, 1, 1, 0, 0])
    assert np.all(compute_distance(c, a, eye.Clusters(c, periodic=True)) == np.array([1, 2, 1]))

    c = np.array([1, 1, 0, 0, 0, 1, 1])
    a = np.array([0, 0, 1, 1, 1, 0, 0])
    assert np.all(compute_distance(c, a, eye.Clusters(c, periodic=True)) == np.array([1, 2, 1]))

    c = np.array([0, 0, 1, 1, 1, 1, 0])
    a = np.array([1, 1, 0, 0, 0, 0, 1])
    assert np.all(compute_distance(c, a, eye.Clusters(c, periodic=True)) == np.array([2, 1, 1]))

# ==================================================================================================
# build histogram
# ==================================================================================================

count_clusters = np.zeros((nx + 1), dtype="int")  # [A]
count_distance = np.zeros((nx + 1, nx + 1), dtype="int")  # [A, distance]
count_size = np.zeros((nx + 1, nx + 1), dtype="int")  # [A, distance]
count_maxsize = np.zeros((nx + 1), dtype="int")  # [A]
norm_clusters = np.zeros((nx + 1), dtype="int")  # [A]
norm_distance = np.zeros((nx + 1), dtype="int")  # [A]

for ifile, file in enumerate(files):

    print(f"({ifile + 1:3d}/{len(files):3d}) {file:s}")

    with h5py.File(file, "r") as data:

        A = data["/sync-A/stored"][...]
        idx0 = data[f"/sync-A/plastic/{np.min(A):d}/idx"][...]
        cracked = np.zeros(idx0.shape, dtype="int")

        for a in A:

            idx = data[f"/sync-A/plastic/{a:d}/idx"][...]
            yielded = (idx != idx0).astype(np.int)
            active = np.where(yielded - cracked == 1, 1, 0)

            if np.sum(cracked) == nx:
                continue

            if np.sum(cracked) > 0:

                clusters = eye.Clusters(cracked, periodic=True)
                sizes = clusters.sizes()
                n, _ = np.histogram(sizes[1:], bins=(nx + 1), range=(0, nx + 1), density=False)

                count_clusters[a] += sizes.size - 1
                count_size[a, :] += n
                count_maxsize[a] += np.max(sizes[1:])
                norm_clusters[a] += 1

            if np.sum(cracked) > 0 and np.sum(active) > 0:

                d = compute_distance(cracked, active, clusters)

                count_distance[a, d] += 1
                norm_distance[a] += 1

            cracked = np.where(yielded + cracked, 1, 0)

# ==================================================================================================
# save data
# ==================================================================================================

with h5py.File(output, "w") as data:

    data["/A"] = np.arange(nx + 1)
    data["/clusters"] = count_clusters / np.where(norm_clusters > 0, norm_clusters, 1)
    data["/distance"] = count_distance / np.where(norm_distance > 0, norm_distance, 1).reshape(
        -1, 1
    )
    data["/size"] = count_size / np.where(norm_clusters > 0, norm_clusters, 1).reshape(-1, 1)
    data["/maxsize"] = count_maxsize / np.where(norm_clusters > 0, norm_clusters, 1)
    data["/norm_clusters"] = norm_clusters
    data["/norm_distance"] = norm_distance

    data["/A"].attrs["desc"] = 'Avalanche areas "A" at which the output is written.'

    data["/clusters"].attrs["desc"] = "Number of clusters [A]. " + 'Normalised by "/norm_clusters".'

    data["/distance"].attrs["desc"] = (
        "Distance between the yielding block(s) and the biggest cluster [A, N]. "
        + 'Normalised by "/norm_distance".'
    )

    data["/size"].attrs["desc"] = (
        "Size of all clusters [A, N]. " + 'Normalised by "/norm_clusters".'
    )

    data["/maxsize"].attrs["desc"] = (
        "Size of the biggest cluster [A]. " + 'Normalised by "/norm_clusters".'
    )

    data["/norm_clusters"].attrs["desc"] = "Normalisation [A]."

    data["/norm_distance"].attrs["desc"] = "Normalisation [A]."
