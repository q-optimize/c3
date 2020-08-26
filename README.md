# c<sup>3</sup>po - Combined Calibration and Characterization by Parameter Optimization

The c<sup>3</sup>po package is intended to close the loop between open-loop control optimization, control pulse calibration, and model-matching based on calibration data.

Currently, only the closed-loop tune-up (calibration) functionality is provided, with pulse parameters optimized using the [CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/) algorithm.  Simulation of the calibration process is possible using [QuTip](http://qutip.org/), although this will soon be replaced by a higher-performance [TensorFlow](http://tensorflow.org) implementation.

c<sup>3</sup>po  provides a simple Python API through which it may integrate with virtually any experimental setup. Such "drivers" are supplied for [LabVIEW](https://www.ni.com/en-us/shop/labview.html) and [Labber](https://labber.org/) -driven experiments, with additional integrations expected soon.

The package is authored by the team at Saarland University. Contact us at [quantum.c3po@gmail.com](mailto://quantum.c3po@gmail.com).

NOTE: This is the 0.1 release. Therefore, expect significant changes as we progress towards v1.0.

## Table of Contents
* [Downloading](#downloading)
* [Installation](#installation)  
* [Usage](#usage)  
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


<a name="usage"><a/>
## Usage
Examples for the usage of the c<sup>3</sup>po package can be found in:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[c3po/c3po/examples/](./c3po/examples/)


### c<sup>3</sup>po for Calibration
An introduction to calibration using c<sup>3</sup>po to send a list of
 parameters to an existing experimental framework (Pycqed, Labber) can be seen
  in the jupyter notebook:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[c3po/c3po/examples/Single_gate_calibration.ipynb](./c3po/examples/Single_gate_calibration.ipynb)

An example for a more low level implementation with the generation of AWG
 signals and communication to a experimental setup is found in:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[c3po/c3po/examples/Single_gate_calibration_labview_setup.ipynb](./c3po/examples/Single_gate_calibration_labview_setup.ipynb)

There, an exemplary code can be seen for calibrating a single qubit X-gate in an experimental
setup using LabVIEW. It is shown how to specifying a pulse and later on how to
theoretically use LabVIEW to communicate with the experiment.

<a name="requirements"><a/>
## Dependencies
- [QuTip](http://qutip.org/)
- [pycma](https://github.com/CMA-ES/pycma)
- [Tensorflow](https://www.tensorflow.org/install)
- [sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html) [sphinx-autoapi](https://sphinx-autoapi.readthedocs.io/en/latest/) [sphinx-rtd-theme](https://github.com/readthedocs/sphinx_rtd_theme) for docs

![C3PO Logo](./C3PO_small.jpg)
