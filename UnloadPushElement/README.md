# UnloadPushElement

Starting for an equilibrium configuration after a system-spanning
event, unload until a certain fixed stress, trigger
at a preselected element,
and compute force equilibrium.

The protocol is as follows:

## job_branchsims.py

The start configurations "push = 0" are taken as the configuration after
a system-spanning event during normal (event-driven) simple shear.
The script `job_branchsis.py` takes care of this,
Several branches per `id=XXX.hdf5` with the same starting configuration
are made in  the files `id=XXX_element=X_inc=X.hdf5`,
whereby `element=X` indicated the element that will be pushed.

## UnloadPushElement

The executable `UnloadPushElement` then applies pushes "push = 0, 1, ..."
at fixed stress, stored in `id=XXX_element=X_inc=X.hdf5`.

*   The event-driven unloading equilibrium configurations are stored.
*   The configuration at mechanical equilibrium, after each sequential push
    is also stored by means of the displacement field to `id=XXX_inc=X.hdf5`
    on the `/push` tree.
    At the same time, the time evolution of each push is written to
    a separate output file `id=XXX_inc=X_push=X.hdf5`.

## job_serial.py

To run on the cluster, `job_serial.py` can generate a single-CPU
job per `id=XXX_element=X_inc=X.hdf5`.
Note that it will automatically skip all files marked as
completed.

## list_status.py

After a run `list_status.py` can be used to collect information
on all files `id=XXX_element=X_inc=X.hdf5` and `id=XXX_element=X_inc=X_push=X.hdf5`.
It writes a file `list_status.yaml` that can then be use e.g.
to copy data.
