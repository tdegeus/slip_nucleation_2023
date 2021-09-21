from setuptools import find_packages
from setuptools import setup

setup(
    name="mycode_front",
    license="MIT",
    author="Tom de Geus",
    author_email="tom@geus.me",
    description="Code for examining front dynamics and sigmac",
    packages=find_packages(),
    use_scm_version={"write_to": "mycode_front/_version.py"},
    setup_requires=["setuptools_scm"],
    entry_points={
        "console_scripts": [
            "JobSerialGroup = mycode_front.slum:cli_serial_group",
            "Run = mycode_front.System:cli_run",
            "Run_generate = mycode_front.System:cli_generate",
            "EnsembleInfo = mycode_front.System:cli_ensembleinfo",
            "PinAndTrigger = mycode_front.PinAndTrigger:cli_main",
            "PinAndTrigger_collect = mycode_front.PinAndTrigger:cli_collect",
            "PinAndTrigger_collect_combine = mycode_front.PinAndTrigger:cli_collect_combine",
        ]
    },
)
