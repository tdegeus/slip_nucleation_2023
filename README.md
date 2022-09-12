# Update to v14

## Dynamics (formerly known as MeasureDynamics)

### Overview

#### Rename functions

-   `MeasureDynamics_average_systemspanning` -> `Dynamics_AverageSystemSpanning`
-   `MeasureDynamics_plot_height` -> `Dynamics_PlotMeshHeight`
-   `MeasureDynamics_run` -> `Dynamics_Run`

## QuasiStatic

### Overview

#### Rename functions

-   `EnsembleInfo` -> `QuasiStatic_EnsembleInfo`
-   `Run_generate` -> `QuasiStatic_Generate`
-   `Run_plot` -> `QuasiStatic_Plot`
-   `RunDynamics_JobAllSystemSpanning` -> `QuasiStatic_MakeJobDynamicsOfSystemSpanning`
-   `RunEventMap_JobAllSystemSpanning` -> `QuasiStatic_MakeJobEventMapOfSystemSpanning`
-   `Run` -> `QuasiStatic_Run`
-   `StateAfterSystemSpanning` -> `QuasiStatic_StateAfterSystemSpanning`
-   `SimulationStatus` -> `QuasiStatic_SimulationStatus`

#### File structure

The file structure is now changed to have different groups:

-   `/param`: Parameters that are the same for all realisations in the ensemble.
-   `/realisation`: Specific realisation settings.
-   `/QuasiStatic`: Output of QuasiStatic loading.
-   `/meta`: Metadata.

Warning:
The `initstate` of a realisation is now:
```python
file["realisation"]["seed"][...] + file["param"]["cusp"]["epsy"][...]
```

Renamed parameters:
```bash
"/coor"  ->  "/param/coor"
"/conn"  ->  "/param/conn"
"/dofs"  ->  "/param/dofs"
"/iip", or "dofsP"  ->  "/param/iip"
"/alpha" ->  "/param/alpha"  # scalar only!
"/eta" ->  "/param/eta"  # scalar only!
"/rho" ->  "/param/rho"  # scalar only!
"/elastic/K" ->  "/param/elastic/K"  # scalar only!
"/elastic/G" ->  "/param/elastic/G"  # scalar only!
"/cusp/elem" ->  "/param/cusp/elem"
"/cusp/K" ->  "/param/cusp/K"  # scalar only!
"/cusp/G" ->  "/param/cusp/G"  # scalar only!
"/cusp/initstate" ->  "/param/cusp/epsy/initstate"  # !!Changed!! See above.
"/cusp/initseq" ->  "/param/cusp/epsy/initseq"
"/cusp/nchunk" ->  "/param/cusp/epsy/nchunk"
"/cusp/eps_offset" ->  "/param/cusp/epsy/weibull/offset"
"/cusp/eps0" ->  "/param/cusp/epsy/weibull/typical"
"/cusp/k" ->  "/param/cusp/epsy/weibull/k"
"/run/dt"  ->  "/param/dt"
"/meta/seed_base"  ->  "/realisation/seed"
"/param/run/epsd/kick"  ->  "/param/cusp/epsy/deps"
```

Deprecated parameters:
```bash
"/elastic/elem"  # recoverable for "/param/conn" and "/param/cusp/elem
"/dofsP"  # was renamed "/iip" as long time ago
```

Still supported parameters, but renamed:
```bash
"/cusp/epsy"  ->  "/param/cusp/epsy"
```

Output:
```bash
"/t"  ->  "/QuasiStatic/inc"  # inc = t / dt
"/disp/..."  ->  "/QuasiStatic/u/..."
"/stored"  # Deprecated!
```
