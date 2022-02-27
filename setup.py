from setuptools import setup, find_packages
import glob
import os

cli_scripts = [
    os.path.basename(path)
    for path in glob.glob(f"{os.path.dirname(__file__)}/cli/cli_*.py")
]

console_scripts = [
    f"""{cli_script.replace(".py", "").replace("cli_", "")} = cli.{cli_script.replace(".py", "")}:main"""
    for cli_script in cli_scripts
]


def parse_requirements(filename):
    """Load requirements from a pip requirements file
    :param filename: str, path to requirements file to load"""
    lineiter = (line.strip() for line in open(filename))
    return [
        line
        for line in lineiter
        if line
        and not line.startswith("#")
        and not line.startswith("--extra-index-url")
        and not line.startswith("git+")
        and not line.startswith("-f")
    ]


setup(
    name="pixray",
    version="0.0.1",
    description="Pixray modulisation",
    author="Paul Asquin",
    packages=find_packages(exclude=["tests"]),
    install_requires=parse_requirements("requirements.txt"),
    entry_points={"console_scripts": console_scripts},
)
