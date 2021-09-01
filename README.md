# Front dynamics - Codes

Codes to measure stress drop and dynamics from pre-run simulations.

<!-- MarkdownTOC -->

- [Codes](#codes)
    - [EnsembleInfo](#ensembleinfo)
    - [CrackEvolution](#crackevolution)
        - [CrackEvolution_stress](#crackevolution_stress)
        - [CrackEvolution_strain](#crackevolution_strain)
        - [CrackEvolution_weak-sync-A_stress](#crackevolution_weak-sync-a_stress)
        - [CrackEvolution_weak-sync-A_strain](#crackevolution_weak-sync-a_strain)
    - [EventEvolution](#eventevolution)
- [Environment](#environment)

<!-- /MarkdownTOC -->

## Codes

### EnsembleInfo

Read information (avalanche size, stress, strain, ...) of an ensemble.

### CrackEvolution

Extract time evolution of a specific push at a fixed stress. This reruns the push and stores the output:

*   At different avalanche radii `A`.
*   At a fixed time from the moment that for the first time `A >= A/2`.

#### CrackEvolution_stress

To extract the average output, the following functions are available:

*   Collect synchronised at different `A` (resulting in a scatter in time `t`):

    -   `collect_sync-A_connect.py`
    -   `collect_sync-A_crack-density.py`
    -   `collect_sync-A_element-components.py`
    -   `collect_sync-A_element.py`
    -   `collect_sync-A_global.py`
    -   `collect_sync-A_plastic.py`
    -   `collect_sync-A_velocity.py`

*   Collect synchronised at different `t` starting from the moment that the avalanches spanned half the system size (resulting in a scatter in `A`):

    -   `collect_sync-t_crack-density.py`
    -   `collect_sync-t_element.py`
    -   `collect_sync-t_global.py`
    -   `collect_sync-t_plastic.py`

#### CrackEvolution_strain

Same as `CrackEvolution_stress`  but for a fixed strain.

#### CrackEvolution_weak-sync-A_stress

Same as `CrackEvolution_stress`  but for reduced storage.

#### CrackEvolution_weak-sync-A_strain

Same as `CrackEvolution_strain`  but for reduced storage.

### EventEvolution

Follow the evolution of an avalanches event by event. This code is similar to [CrackEvolution](#crackevolution) but with different storage.

## Environment

For each code the environment can be entirely set using *Conda*. In particular, each code has a file `environment.yaml` which contains the necessary dependencies.

*   Create an environment based on the environment file:

    ```
    conda env create --name NAME --file FILE
    ```

*   Update an environment based on the environment file:

    ```
    source activate NAME
    conda env update --file FILE
    ```

    or

    ```
    conda env update --name NAME --file FILE
    ```
