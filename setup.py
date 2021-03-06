from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.rst")) as readme_file:
    readme = readme_file.read()

with open(path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line
        for line in requirements_file.read().splitlines()
        if not line.startswith("#")
    ]

EXTRAS_REQUIRE = {
    "tests": ["black", "flake8", "invoke", "pytest", "pytest-cov"],
    "docs": [
        "invoke",
        "matplotlib",
        "numpydoc",
        "sphinx",
        "sphinx-gallery",
        "sphinx_rtd_theme",
        "sphinx-copybutton",
    ],
}


setup(
    name="dagging",
    version="0.2.dev",
    description="Python package implementing the dagging method",
    long_description=readme,
    author="Christos Aridas",
    author_email="ichkoar@gmail.com",
    url="https://github.com/chkoar/dagging",
    packages=find_packages(exclude=["docs", "tests"]),
    zip_safe=False,  # the package can run out of an .egg file
    entry_points={
        "console_scripts": [
            # 'some.module:some_function',
        ]
    },
    include_package_data=True,
    package_data={
        "dagging": [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=requirements,
    extras_require=EXTRAS_REQUIRE,
    license="BSD (3-clause)",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
)
