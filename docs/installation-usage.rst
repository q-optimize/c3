Installation & Usage
====================

Installation
************

The easiest way to use the c3po package is through the installation with pip, in developer mode.

Place the source files in the directory of your choice, and then run
::

    pip install -e <c3po source directory>


Adding the -e specifies the developer option and will result in the source directory being linked into pip's index 
(and nothing will be downloaded, except any required dependencies, such as QuTip and pycma).

To update at any point, simply update the files in the .

Attention: As explained above, this does only link the c3po folder to your local python packages. 
Deleting the c3po folder does therefore also result in the deletion of the c3po package.


Usage
*****

Examples for the usage of the c3po package can be found in::

    c3po/c3po/examples/

**c3po for Calibration**

An introduction to calibration using c3po to send a list of parameters to an 
existing experimental framework (Pycqed, Labber) can be seen in the jupyter notebook::

    c3po/c3po/examples/Single_gate_calibration.ipynb

An example for a more low level implementation with the generation of AWG signals and communication to a experimental setup is found in::

    c3po/c3po/examples/Single_gate_calibration_labview_setup.ipynb

There, an exemplary code can be seen for calibrating a single qubit X-gate in an experimental setup using LabVIEW. 
It is shown how to specifying a pulse and later on how to theoretically use LabVIEW to communicate with the experiment.