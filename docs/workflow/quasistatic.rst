QuasiStatic
-----------

Stating point
:::::::::::::

Quasistatic, event-driven, simulations.

1.  Generate realisations [ensemble]

    :ref:`QuasiStatic_Generate`

2.  Run event-driven quasistatic simulations of one realisation [realisation]

    :ref:`QuasiStatic_Run`

3.  Collect basic output data of ensemble of realisations [ensemble]

    :ref:`QuasiStatic_EnsembleInfo`

Post-process
::::::::::::

Post-process the output of the quasistatic simulations:

-   State after system spanning events [ensemble]

    :ref:`QuasiStatic_StateAfterSystemSpanning`

-   Basic plotting [realisation or ensemble]

    :ref:`QuasiStatic_Plot`

Dynamics
::::::::

Analyse dynamics of event(s)

1.  Branch from realisation(s) [ensemble]

    :ref:`QuasiStatic_MakeJobDynamicsOfSystemSpanning`

2.  Re-run single event [realisation]

    :ref:`Dynamics_Run`

3.  Collect average dynamics from several events [ensemble]

    :ref:`Dynamics_AverageSystemSpanning`

Dynamics - high speed observations
::::::::::::::::::::::::::::::::::

Analyse dynamics of event(s) at high speed.
This collects only the averages and the information at some sensors.
To plot sensor locations: :ref:`Dynamics_PlotMeshHeight`

1.  Branch from realisation(s) [ensemble]

    :ref:`QuasiStatic_MakeJobDynamicsOfSystemSpanning`

2.  Re-run single event [realisation]

    :ref:`Dynamics_RunHighFrequency`

EventMap
::::::::

Analyse the sequences failures of one event

1.  Branch from realisation(s) [ensemble]

    :ref:`QuasiStatic_MakeJobEventMapOfSystemSpanning`

1.  Re-run event, store time and position of each failure [realisation]

    :ref:`EventMap_Run`

2.  Extract basic info [ensemble]

    :ref:`EventMap_Info`

Trigger
:::::::

Branch to trigger at different stress

1.  Branch quasistatic simulations [ensemble]

    :ref:`Trigger_JobDeltaSigma`

2.  Trigger and minimise [realisation]

    :ref:`Trigger_Run`

3.  Collect basic output data [ensemble]

    :ref:`Trigger_EnsembleInfo`

4.  Group many many triggers (for data compression) [ensemble]

    :ref:`Trigger_EnsemblePack`
