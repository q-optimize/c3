# c3po
Combined Calibration and Characterization by Parameter Optimization

The c3po package utilizes the at its current stage the functionality provided
by the [qutip](http://qutip.org/) package to implement the task of automated
calibration with parameter optimization. The parameter optimization is hereby
done with the [CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/)
algorithm.  

The aim is also to provide a package that enables an easy usage in
pre-existing experimental setups consisting of i.e.
[LabVIEW](https://www.ni.com/en-us/shop/labview.html),
[Labber](https://labber.org/), ...

Remark:  
As the project is in a very early stage of the development the name might
change in future releases.

## Table of Contents
[Installation](#installation)  
[Usage](#usage)  
[Requirements](#requirements)  
[Misc](#misc)  

<a name="installation"/>
## Installation
In the current state of the project the easiest way to use the c3po package is
through the installation with [pip](https://pypi.org/project/pip/).

### Developer installation
To install the package with pip simply run
```
pip install -e <source directory>
```
Adding the -e specifies the developer option and will only link to the source.
This way you don't have to reinstall after changes in the code.

**Attention:** As explained above, this does only link the c3po folder to your
local python packages. Deleting the c3po folder does therefore also result in
the deletion of the c3po package.

#### Example
##### Linux
After the download copy the c3po.zip file to the location you want the package
to be installed. Unzip the package and open a terminal. Navigate in the
terminal to the location you just extracted the .zip file in and run the
following command within our terminal:
```
pip install -e c3po
```
<a name="usage"/>
## Usage
Examples for the usage of the c3po package can be found in:
```

c3po/c3po/examples/

```
### c3po for Calibration
An introduction to calibration using c3po can be seen in the jupyter notebook:
```
 c3po/c3po/examples/Single_gate_calibration_labview_setup.ipynb
```
There, an exemplary code can be seen for calibrating a X-gate in an experimental
setup using LabVIEW. It is shown how to specifying a pulse and later on how to
theoretically use LabVIEW to communicate with the experiment.

<a name="requirements"/>
## Requirements
- [qutip](http://qutip.org/)
- [pycma](https://github.com/CMA-ES/pycma)

<a name="misc"/>
## Misc
### Notation
- [Hamiltonian notation](http://qutip.org/docs/latest/guide/dynamics/dynamics-time.html)
