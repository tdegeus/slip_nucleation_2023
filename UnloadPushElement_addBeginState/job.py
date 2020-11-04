import os
import re
import subprocess

files = sorted(list(filter(None, subprocess.check_output(
    "find . -iname 'id*.hdf5'", shell=True).decode('utf-8').split('\n'))))

files = [file for file in files if len(file.split('push')) == 1]

for file in files:
    print(file)
    basename = os.path.splitext(file)[0]
    print(subprocess.check_output("./UnloadPushElement_addBeginState --input {0:s} --output {1:s}".format(file, basename), shell=True).decode())
