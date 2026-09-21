import re
import sys

from setuptools import find_packages, setup


def read_version():
    """Read __version__ without importing the package (which needs torch)."""
    with open("pairwise/__init__.py") as f:
        return re.search(r'^__version__ = "([^"]+)"', f.read(), re.M).group(1)


def parse_requirements(requirements):
    with open(requirements) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

NAME = "pairwise-synergy"
VERSION = read_version()
AUTHOR = "Chengqi Xu"
DESCRIPTION = (
    "PAIRWISE: deep learning-based prediction of effective personalized "
    "drug combinations in cancer."
)
URL = "https://github.com/Mew233/pairwise"

REQUIRED_PYTHON_VERSION = (3, 10)
PACKAGES = find_packages(
    exclude=["paper", "paper.*", "example", "baseline", "misc", "SI"]
)

INSTALL_DEPENDENCIES = parse_requirements("./requirements.txt")
TEST_DEPENDENCIES = ["pytest"]
EXTRA_DEPENDENCIES = {"dev": ["pytest"]}

if sys.version_info < REQUIRED_PYTHON_VERSION:
    sys.exit("Python >= 3.10 is required. Your version:\n" + sys.version)

setup(
    name=NAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    version=VERSION,
    author=AUTHOR,
    license="MIT",
    packages=PACKAGES,
    include_package_data=True,
    package_data={"pairwise": ["configs/*.json"]},
    install_requires=INSTALL_DEPENDENCIES,
    tests_require=TEST_DEPENDENCIES,
    extras_require=EXTRA_DEPENDENCIES,
    entry_points={
        "console_scripts": [
            "pairwise-train = pairwise.main:main",
        ]
    },
    python_requires=">=3.10",
)
