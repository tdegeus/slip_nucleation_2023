from setuptools import find_packages
from setuptools import setup

setup(
    name="mycode_front",
    license="MIT",
    author="Tom de Geus",
    author_email="tom@geus.me",
    description="Code for examining front dynamics",
    packages=find_packages(),
    use_scm_version={"write_to": "mycode_front/_version.py"},
    setup_requires=["setuptools_scm"],
    entry_points={
        "console_scripts": [
            "PinAndTrigger = mycode_front.PinAndTrigger:cli_main",
            "PinAndTrigger_collect = mycode_front.PinAndTrigger:cli_collect",
        ]
    },
)
