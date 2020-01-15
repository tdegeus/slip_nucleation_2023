# Front dynamics - Codes

Codes to measure stress drop and dynamics from pre-run simulations.

## Codes

### EnsembleInfo

Read information (avalanche size, stress, strain, ...) of an ensemble.

### AvalancheEvolution_stress

Extract time evolution of a specific push. This reruns the push and store the output at different avalanche radii `A`.

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
