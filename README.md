# C3 - An integrated tool-set for Control, Calibration and Characterization

[![codecov](https://codecov.io/gh/q-optimize/c3/branch/dev/graph/badge.svg)](https://codecov.io/gh/q-optimize/c3)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/q-optimize/c3.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/q-optimize/c3/context:python)
<a href="https://codeclimate.com/github/q-optimize/c3/maintainability"><img src="https://api.codeclimate.com/v1/badges/a090831b106f863dc223/maintainability" /></a>
[![Build and Test](https://github.com/q-optimize/c3/actions/workflows/build_package.yml/badge.svg)](https://github.com/q-optimize/c3/actions/workflows/build_package.yml)
[![Documentation Status](https://readthedocs.org/projects/c3-toolset/badge/?version=latest)](https://c3-toolset.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version fury.io](https://badge.fury.io/py/c3-toolset.svg)](https://pypi.python.org/pypi/c3-toolset/)
[![PyPI license](https://img.shields.io/pypi/l/c3-toolset.svg)](https://pypi.python.org/pypi/c3-toolset/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/c3-toolset.svg)](https://pypi.python.org/pypi/c3-toolset/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/q-optimize/c3/dev)

The C3 package is intended to close the loop between open-loop control optimization, control pulse calibration, and model-matching based on calibration data.

## Installation

```bash
pip install c3-toolset
```

If you want to try out the bleeding edge (possibly buggy) version under development:

```bash
pip install c3-toolset-nightly
```

## Usage

C3  provides a simple Python API through which it may integrate with virtually any experimental setup.
Contact us at [c3@q-optimize.org](mailto://quantum.c3po@gmail.com).

The paper introducing C3 as a concept can be found on the [arxiv](https://arxiv.org/abs/2009.09866).

Documentation is available [here](https://c3-toolset.readthedocs.io) on RTD.

Examples are available in the `examples/` directory and can also be run online using the `launch|binder` badge above.

If you wish to contribute, please check out the issues tab and also the `CONTRIBUTING.md` for useful resources.

The source code is available on Github at [https://github.com/q-optimize/c3](https://github.com/q-optimize/c3).

## Citation

If you use `c3-toolset` in your research, please cite it as below:

```
@article{Wittler2021,
   title={Integrated Tool Set for Control, Calibration, and Characterization of Quantum Devices Applied to Superconducting Qubits},
   volume={15},
   DOI={10.1103/physrevapplied.15.034080},
   number={3},
   journal={Physical Review Applied},
   author={Wittler, Nicolas and Roy, Federico and Pack, Kevin and Werninghaus, Max and Saha Roy, Anurag and Egger, Daniel J. and Filipp, Stefan and Wilhelm, Frank K. and Machnes, Shai},
   year={2021},
   month={Mar}
}
```