# PushRecursiveQuasiThermal

With a fixed time interval mimic thermal fluctuations that can 
jump a local energy barrier. 
In particular, the height of the barrier "W" to reach the yield surface is measured 
for each integration point
(approximately, but phenomenologically correct, for a simple shear perturbation).
The element is then triggered by comparing the maximum barrier in the element
to a Boltzmann weight. 
In particular, the element is triggered with a probability exp(- W / (kB T)).
This is continued until the macroscopic stress reaches a target value.
Finally, mechanical equilibrium is sought.

<!-- MarkdownTOC -->

- [PushRecursiveQuasiThermal](#pushrecursivequasithermal)
- [job_branchsims*](#job_branchsims)
- [job_serial*](#job_serial)
- [mechanisms*](#mechanisms)

<!-- /MarkdownTOC -->

## PushRecursiveQuasiThermal

The executable `PushRecursiveQuasiThermal` then does the quasi-thermal triggering, 
until a target stress is reached. 
The final, equilibrium state, is stored to the input file
`id=XXX_inc=XX_target=sigc-Xd6_kBT=XXX.hdf5`.

With a second command-line argument, the event-driven output can be written.

## job_branchsims*

The start configurations "push = 0" are taken as the configuration after
a system-spanning event during normal (event-driven) simple shear.
The script `job_branchsims.py` takes care of this.
It generates per `id=XXX.hdf5`: `id=XXX_inc=XX_target=sigc-Xd6_kBT=XXX.hdf5`.
The temperature, target stress, etc. are stored as variables.

>   The yield strains are extended to allow for a finite shear. 

*   `job_branchsims.py`: Generate for various temperatures.
*   `job_branchsims_preparation-only.py`: Generate with only one temperature.

## job_serial*

Generate a single-CPU job per `id*.hdf5`. 

*   `job_serial.py`: Run with event-driven output file.
*   `job_serial_no-output.py`: Run with no event-driven output file.

## mechanisms*

Plot event-driven output.
