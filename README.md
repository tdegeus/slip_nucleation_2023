# Front dynamics - Codes

Codes to measure stress drop and dynamics from pre-run simulations.

## Codes

### EnsembleInfo

Read information (avalanche size, stress, strain, ...) of an ensemble.

### AvalancheEvolution_stress

Extract time evolution of a specific push at a fixed stress. This reruns the push and stores the output at different avalanche radii `A`. It is meant mostly to study those events that did not grow system spanning (and did not nucleate fracture). To study 'fracture' please use `CrackEvolution_stress` and `CrackEvolution_strain`.

### CrackEvolution_stress

Extract time evolution of a specific push at a fixed stress. This reruns the push and stores the output:

*   At different avalanche radii `A`.
*   At a fixed time from the moment that for the first time `A >= A/2`.

To extract the average output, the following functions are available:

*   Collect synchronised at different `A` (resulting in a scatter in time `t`):

    -   `collect_sync-A_element-components.py`
    -   `collect_sync-A_element.py`
    -   `collect_sync-A_global.py`
    -   `collect_sync-A_plastic.py`

*   Collect synchronised at different `t` starting from the moment that the avalanches spanned half the system size (resulting in a scatter in `A`):

    -   `collect_sync-t_element.py`
    -   `collect_sync-t_global.py`
    -   `collect_sync-t_plastic.py`

### CrackEvolution_strain

Same as `CrackEvolution_stress`  but for a fixed strain.

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
