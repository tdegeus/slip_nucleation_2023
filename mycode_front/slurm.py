import argparse
import os
import sys
import textwrap

import GooseSLURM
import numpy as np

default_condabase = "code_velocity"

entry_points = dict(
    cli_serial_group="JobSerialGroup",
)

slurm_defaults = dict(
    account="pcsl",
)


def snippet_export_omp_num_threads(ncores=1):
    """
    Return code to set OMP_NUM_THREADS
    :return: str
    """
    return f"# Set number of cores to use\nexport OMP_NUM_THREADS={ncores}"


def snippet_load_conda(condabase: str = default_condabase):
    """
    Return code to load the Conda environment.
    :param condabase: Base name of the Conda environment, appended '_E5v4' and '_s6g1'".
    :return: str
    """

    ret = [
        textwrap.dedent(
            """
            # --- Functions to load/clean Conda environment ---

            # check if a list contains a string
            # param: string list
            contains ()
            {
                local e match="$1"
                shift
                for e; do [[ "$e" == "$match" ]] && return 0; done
                return 1
            }

            # print current environment
            conda_echo_env()
            {
                echo "conda env: $CONDA_DEFAULT_ENV"
            }

            # deactivate all environments, except base
            conda_deactivate ()
            {
                while [[ "$CONDA_DEFAULT_ENV" != "base" ]]; do
                    conda deactivate
                done
                conda_echo_env
            }

            # activate environment starting from base
            # param: environment_name
            conda_activate ()
            {
                conda_deactivate
                conda activate "$1"
                conda_echo_env
            }

            # activate the first present environment
            # param: environment_name ...
            conda_activate_first_existing ()
            {
                conda_deactivate

                ENVS=($(conda env list | awk '{print $1}'))

                for env in "$@"
                do
                    if contains "$env" "${ENVS[@]}"; then
                        conda activate "$env"
                        conda_echo_env
                        return 0
                    fi
                done

                echo "ERROR: Failed to activate any environment"
                exit 1
            }

            # remove and create environment(s)
            # param: environment_name ...
            conda_clean ()
            {
                conda_deactivate

                ENVS=($(conda env list | awk '{print $1}'))

                for env in "$@"
                do
                    if contains "$env" "${ENVS[@]}"; then
                        echo "conda remove -n $env"
                        conda env remove -n "$env"
                    fi
                    echo "conda create -n $env"
                    mamba create -n "$env" -y
                done
            }

            # get name postfix for current SYS_TYPE (return empty string if SYS_TYPE is unknown)
            function get_simd_envname()
            {
                if [[ "${SYS_TYPE}" == *E5v4* ]]; then
                    echo "_E5v4"
                elif [[ "${SYS_TYPE}" == *s6g1* ]]; then
                    echo "_s6g1"
                elif [[ "${SYS_TYPE}" == *S6g1* ]]; then
                    echo "_s6g1"
                else
                    echo ""
                fi
            }

            # ---
            """
        )
    ]

    ret += ["# Activate hardware optimised environment (or fallback environment)"]
    ret += [f'conda_activate_existing "{condabase}$(get_simd_envname)" "{condabase}"']
    ret += []

    return "\n".join(ret)


def snippet_flush(cmd):
    """
    Return code to run a command and flush the buffer of stdout.
    :param cmd: The command.
    :return: str
    """
    return "stdbuf -o0 -e0 " + cmd


def script_exec(cmd, omp_num_threads=True, conda=True, flush=True):
    """
    Return code to execute a command.
    Optionally a number of extra commands are run before the command itself, see options.
    Defaults of the underlying functions can be overwritten by passing a tuple or dictionary.
    The option can be skipped by specifying ``None`` or ``False``.

    For example::
        slurm.script_exec(cmd, conda=dict(condabase="my"))
        slurm.script_exec(cmd, conda=dict(condabase="my"))

    :param cmd: The command.
    :param omp_num_threads: Number of cores to use (see snippet_export_omp_num_threads()).
    :param conda: Load conda environment (see defaults of snippet_load_conda()).
    :param flush: Flush the buffer of stdout.
    :return: str
    """

    ret = []

    for opt, func in zip(
        [omp_num_threads, conda],
        [snippet_export_omp_num_threads, snippet_load_conda],
    ):
        if opt is True:
            ret += [func(), ""]
        elif opt is not None and opt is not False:
            if type(opt) == dict:
                ret += [func(**opt), ""]
            else:
                ret += [func(*opt), ""]

    if flush:
        ret += ["# --- Run ---", "", snippet_flush(cmd), ""]
    else:
        ret += ["# --- Run ---", "", cmd, ""]

    return "\n".join(ret)


def serial_group(
    commands: list[str],
    basename: str,
    group: int,
    outdir: str = os.getcwd(),
    sbatch: dict = {},
    omp_num_threads=True,
    conda=True,
    flush=True,
):
    """
    Group a number of commands per job-script.

    :param commands: List of commands.
    :param basename: Base-name of the job-scripts (and their log-scripts),
    :param group: Number of commands to group per job-script.
    :param outdir: Directory where to write the job-scripts (nothing in changed for the commands).
    :param sbatch: Job options.
    :param omp_num_threads: Number of cores to use (see snippet_export_omp_num_threads()).
    :param conda: Load conda environment (see defaults of snippet_load_conda()).
    :param flush: Flush the buffer of stdout for each commands.
    """

    if len(commands) == 0:
        return

    assert "job-name" not in sbatch
    assert "out" not in sbatch
    sbatch.setdefault("nodes", 1)
    sbatch.setdefault("ntasks", 1)
    sbatch.setdefault("cpus-per-task", 1)
    sbatch.setdefault("time", "24h")
    sbatch.setdefault("account", slurm_defaults["account"])
    sbatch.setdefault("partition", "serial")

    if flush:
        commands = [snippet_flush(cmd) for cmd in commands]

    chunks = int(np.ceil(len(commands) / float(group)))
    devided = np.array_split(commands, chunks)
    njob = len(devided)
    fmt = str(int(np.ceil(np.log10(njob))))

    for g, selection in enumerate(devided):

        command = script_exec(
            "\n".join(selection),
            omp_num_threads=omp_num_threads,
            conda=conda,
            flush=False,
        )

        jobname = ("{0:s}_{1:0" + fmt + "d}-of-{2:d}").format(basename, g + 1, njob)
        sbatch["job-name"] = jobname
        sbatch["out"] = jobname + "_%j.out"

        with open(os.path.join(outdir, jobname + ".slurm"), "w") as file:
            file.write(GooseSLURM.scripts.plain(command=command, **sbatch))


def cli_serial_group(cli_args=None):
    """
    Job-script to run commands.
    """

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent(cli_serial_group.__doc__),
    )

    parser.add_argument(
        "files",
        nargs="*",
        type=str,
        help="Files to run",
    )

    parser.add_argument(
        "-o", "--outdir", type=str, default=os.getcwd(), help="Output directory"
    )

    parser.add_argument(
        "-c",
        "--command",
        type=str,
        help="Command to use",
    )

    parser.add_argument(
        "-n",
        "--group",
        type=int,
        default=1,
        help="Number of commands to group in a single job",
    )

    parser.add_argument(
        "-w",
        "--time",
        type=str,
        default="24h",
        help="Walltime to allocate for the job",
    )

    args = parser.parse_args(cli_args)

    assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])

    files = [os.path.relpath(file, args.outdir) for file in args.files]
    commands = [f"{args.command} {file}" for file in files]
    serial_group(
        commands,
        basename=args.command,
        group=args.group,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )
