# c<sup>3</sup>po - Combined Calibration and Characterization by Parameter Optimization

The c<sup>3</sup>po package is intended to close the loop between open-loop control optimization, control pulse calibration, and model-matching based on calibration data.

Currently, only the closed-loop tune-up (calibration) functionality is provided, with pulse parameters optimized using the [CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/) algorithm.  Simulation of the calibration process is possible using [QuTip](http://qutip.org/), although this will soon be replaced by a higher-performance [TensorFlow](http://tensorflow.org) implementation.

c<sup>3</sup>po  provides a simple Python API through which it may integrate with virtually any experimental setup. Such "drivers" are supplied for [LabVIEW](https://www.ni.com/en-us/shop/labview.html) and [Labber](https://labber.org/) -driven experiments, with additional integrations expected soon.

The package is authored by the team at Saarland University. Contact us at [c3@q-optimize.org](mailto://quantum.c3po@gmail.com).

Documentation is available [here](https://c3-toolset.readthedocs.io).

NOTE: This is the 0.1 release. Therefore, expect significant changes as we progress towards v1.0.

## Table of Contents
* [Downloading](#downloading)
* [Installation](#installation)  
* [Requirements](#requirements)  
* [Misc](#misc)  

<a name="downloading"><a/>
## Downloading
Until the project reaches the v1.0 release, source code will be provided upon request by the project team at Saarland University. Please contact us at [quantum.c3po@gmail.com](mailto://quantum.c3po@gmail.com).

<a name="installation"><a/>
## Installation (developer mode)

The easiest way to use the c<sup>3</sup>po package is through the installation with [pip](https://pypi.org/project/pip/), in developer mode.

Place the source files in the directory of your choice, and then run
```
pip install -e <c3po source directory>
```
Adding the -e specifies the developer option and will result in the source directory being linked into pip's index (and nothing will be downloaded, except any required dependencies, such as [QuTip](http://qutip.org/) and [pycma](https://github.com/CMA-ES/pycma)).

To update c<sup>3</sup>po at any point, simply update the files in the <c3po source directory>.


**Attention:** As explained above, this does only link the c<sup>3</sup>po folder to your
local python packages. Deleting the c<sup>3</sup>po folder does therefore also result in
the deletion of the c<sup>3</sup>po package.


<a name="requirements"><a/>
## Dependencies
- [QuTip](http://qutip.org/)
- [pycma](https://github.com/CMA-ES/pycma)
