import argparse
import os
import sys
import textwrap

import GooseSLURM
import numpy as np

default_condabase = "code_velocity"
default_condaexec = "~/miniconda3/etc/profile.d/conda.sh"


def script_echo_jobid():
    """
    Return code to echo the job-id.
    :return: str
    """
    return textwrap.dedent(
        """
        # print jobid
        echo "SLURM_JOBID = ${{SLURM_JOBID}}"
        echo ""
        """
    )


def script_export_omp_num_threads(ncores=1):
    """
    Return code to set OMP_NUM_THREADS
    :return: str
    """
    return textwrap.dedent(
        f"""
        # set the number of cores to use by OMP
        export OMP_NUM_THREADS={ncores}
        """
    )


def script_load_conda(
    condabase: str = default_condabase, condaexec: str = default_condaexec
):
    """
    Return code to load the Conda environment.
    :param condabase: Base name of the Conda environment, appended '_E5v4' and '_s6g1'".
    :param condaexec: Path of the Conda executable.
    :return: str
    """

    return textwrap.dedent(
        f"""
        # load conda environment
        source {condaexec}

        if [[ "${{SYS_TYPE}}" == *E5v4* ]]; then
            conda activate {condabase}_E5v4
        elif [[ "${{SYS_TYPE}}" == *s6g1* ]]; then
            conda activate {condabase}_s6g1
        elif [[ "${{SYS_TYPE}}" == *S6g1* ]]; then
            conda activate {condabase}_s6g1
        else
            echo "Unknown SYS_TYPE ${{SYS_TYPE}}"
            exit 1
        fi
        """
    )


def script_flush(cmd):
    """
    Return code to run a command and flush the buffer of stdout.
    :param cmd: The command.
    :return: str
    """
    return "stdbuf -o0 -e0 " + cmd


def script_exec(cmd, jobid=True, omp_num_threads=True, conda=True, flush=True):
    """
    Return code to execute a command.
    Optionally a number of extra commands are run before the command itself, see options.
    Defaults of the underlying functions can be overwritten by passing a tuple or dictionary.
    The option can be skipped by specifying ``None`` or ``False``.

    For example::
        slurm.script_exec(cmd, conda=dict(condabase="my"))
        slurm.script_exec(cmd, conda=dict(condabase="my"))

    :param cmd: The command.
    :param jobjd: Echo the jobid (see script_echo_jobid()).
    :param omp_num_threads: Number of cores to use (see script_export_omp_num_threads()).
    :param conda: Load conda environment (see defaults of script_load_conda()).
    :param flush: Flush the buffer of stdout.
    :return: str
    """

    ret = []

    for opt, func in zip(
        [jobid, omp_num_threads, conda],
        [script_echo_jobid, script_export_omp_num_threads, script_load_conda],
    ):
        if opt is True:
            ret += [func()]
        elif opt is not None and opt is not False:
            if type(opt) == dict:
                ret += [func(**opt)]
            else:
                ret += [func(*opt)]

    ret += []

    if flush:
        ret += [script_flush(cmd)]
    else:
        ret += [cmd]

    return "\n".join(ret)


def serial_group(
    commands: list[str],
    basename: str,
    group: int,
    outdir: str = os.getcwd(),
    sbatch: dict = {},
    jobid=True,
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
    :param jobjd: Echo the jobid (see script_echo_jobid()).
    :param omp_num_threads: Number of cores to use (see script_export_omp_num_threads()).
    :param conda: Load conda environment (see defaults of script_load_conda()).
    :param flush: Flush the buffer of stdout for each commands.
    """

    assert "job-name" not in sbatch
    assert "out" not in sbatch
    sbatch.setdefault("nodes", 1)
    sbatch.setdefault("ntasks", 1)
    sbatch.setdefault("cpus-per-task", 1)
    sbatch.setdefault("time", "24h")
    sbatch.setdefault("account", "pcsl")
    sbatch.setdefault("partition", "serial")

    if flush:
        commands = [script_flush(cmd) for cmd in commands]

    ngroup = int(np.ceil(len(commands) / group))
    fmt = str(int(np.ceil(np.log10(ngroup))))

    for g in range(ngroup):

        ii = g * group
        jj = (g + 1) * group
        c = commands[ii:jj]
        command = script_exec(
            "\n".join(c),
            jobid=jobid,
            omp_num_threads=omp_num_threads,
            conda=conda,
            flush=False,
        )

        jobname = ("{0:s}_{1:0" + fmt + "d}-of-{2:d}").format(basename, g + 1, ngroup)
        sbatch["job-name"] = jobname
        sbatch["out"] = jobname + ".out"

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
