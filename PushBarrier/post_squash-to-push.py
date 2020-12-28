import sys
import subprocess
import h5py

files = sorted(list(filter(None, subprocess.check_output(
    "find . -iname '*push*hdf5'", shell=True).decode('utf-8').split('\n'))))

for push in files:
    base = push.split('_push')[0] + '.hdf5'

    with h5py.File(base, 'r') as data:

        with h5py.File(push, 'r') as push_data:
            assert data['/uuid'][...] == push_data['/meta/uuid'][...]
            assert data['/git/PushBarrier'][...] == push_data['/git/PushBarrier'][...]

        with h5py.File(push, 'a') as push_data:

            push_data['/t'] = data['/t'][...][:2]
            push_data['/sigd'] = data['/sigd'][...][:2]
            push_data['/disp/0'] = data['/disp/0'][...]
            push_data['/disp/1'] = data['/disp/1'][...]
            push_data['/meta/trigger/i0'] = data['/trigger/i'][...]

            if 'failed_push' in data:
                push_data['/failed_push/element'] = data['/failed_push/element'][...]
                push_data['/failed_push/inc'] = data['/failed_push/inc'][...]
