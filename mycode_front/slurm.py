import textwrap

default_condabase = "code_velocity"
default_condaexec = "~/miniconda3/etc/profile.d/conda.sh"
default_jobbase = "velocity"

def script_echo_jobid():
    """
    Return code to echo the job-id.
    :return: str
    """
    return textwrap.dedent(f"""
    # print jobid
    echo "SLURM_JOBID = ${{SLURM_JOBID}}"
    echo ""
    """)

def script_export_omp_num_threads(ncores=1):
    """
    Return code to set OMP_NUM_THREADS
    :return: str
    """
    return textwrap.dedent(f"""
    # set the number of cores to use by OMP
    export OMP_NUM_THREADS={ncores}
    """)

def script_load_conda(condabase: str=default_condabase, condaexec: str=default_condaexec):
    """
    Return code to load the Conda environment.
    :param condabase: Base name of the Conda environment, appended '_E5v4' and '_s6g1'".
    :param condaexec: Path of the Conda executable.
    :return: str
    """

    return textwrap.dedent(f"""
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
    """)


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

    for opt, func in zip([jobid, omp_num_threads, conda], [script_echo_jobid, script_export_omp_num_threads, script_load_conda]):
        if opt is True:
            ret += [func()]
        elif opt is not None and opt is not False:
            if type(opt) == dict:
                ret += [func(**opt)]
            else:
                ret += [func(*opt)]

    if flush:
        ret += [script_flush(cmd)]
    else:
        ret += [cmd]

    return "\n".join(ret)

