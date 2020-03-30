# Optimization of DLA structures using inverse design

This repository contains a tutorial on the optimization of a dielectric laser acceleration (DLA) structure with the technique of inverse design.


## Getting started with Python

If you have not used Python before, I highly recommend you to use Anaconda as it comes with many packages and offers a set of IDEs. In that case, visit [ANACONDA](https://www.anaconda.com/distribution/) and install the latest Python 3 version (currently Python 3.7, preferentially the 64-bit version).

For any further installations with Anaconda, you will want to use Anaconda Prompt. You can find it by searching your programs for "Anaconda Prompt".

For experienced Python users, you may use the cmd, but make sure that you have [Jupyter](https://jupyter.org/) Notebook installed. If you want to write you own Pyhton scripts, I recommend to use an IDE (integrated development environment), such as [Spyder](https://www.spyder-ide.org/), which combines a text editor, console and debugger. Both Jupyther and Spyder are included in the Anaconda distribution.

## Installation of ceviche

**[Ceviche](https://github.com/fancompute/ceviche)** is a finite difference frequency domain (FDFD) and time domain (FDTD) package for solving Maxwell's equations. It was developed by Tyler W Hughes, Ian AD Williamson and Momchil Minkov in [Shanhui Fan's group](https://web.stanford.edu/group/fan/) at Stanford University. We will use `ceviche` as the primary tool to simulate and optimize optical devices.

The source code of `ceviche` is freely available on GitHub and there are multiple ways to install it on your computer. I recommend to download it as a zip archive by visiting [https://github.com/fancompute/ceviche](https://github.com/fancompute/ceviche), clicking on the green "Clone or download" button in the top right corner of the GitHub repository page and then selecting "Download ZIP" from the drop down menu. After that, you can extract the files and save them in a new folder, e.g., called "InverseDesign".

In order to use the material of this [tutorial](https://github.com/uhaeusler/tutorial_invDesign_DLA), you can proceed likewise and save it in the same folder as above. Now the folder "InverseDesign" should contain a folder "ceviche-master" and a folder "tutorial_invDesign_DLA-master".

Alternatively, if you have `git` installed, you can enter the following command in a terminal to clone the repositories:

    git clone https://github.com/fancompute/ceviche.git
    git clone https://github.com/uhaeusler/tutorial_invDesign_DLA.git

Finally, there is a bunch of packages that we need to install before we can run all scripts of this repository.

**[HIPS autograd](https://github.com/HIPS/autograd)** is an automatic differentiation framework with a [Numpy](https://numpy.org/)-like API, which we use for the inverse design optimization algorithm.

**[pyMKL](https://pypi.org/project/pyMKL/)** provides an interface to the [PARDISO](https://www.pardiso-project.org/) sparse solver, which can speed up our simulations compared to the standard SciPy sparse linear solver routines.

**[scikit-image](https://scikit-image.org/)** offers useful functions for drawing shapes into 2D arrays, which we apply to define geometrical features on our photonic device.

### Anaconda Prompt
If you are using Anaconda, open Anaconda Prompt and type

    conda install -c conda-forge autograd
    conda install -c conda-forge pymkl
    conda install -c conda-forge scikit-image

### CMD
If you are not using Anaconda, you might need to install some more packages, which otherwise come together with the Anaconda distribution. For this, open the cmd and type

    pip install autograd
    pip install matplotlib
    pip install numpy
    pip install pymkl
    pip install scikit-image
    pip install scipy

### PIP INSTALL CEVICHE

If you followed the instructions above, you are now ready to use ceviche. Let me point out that there is actually another way of accessing `ceviche`, which actually qualifies for the term "installation". For this, we open Anaconda Prompt (or cmd if you are not using conda) and type

    pip install ceviche

This command will automatically install some other Python packages (autograd, matplotlib, numpy, pyMKL, scipy), which are needed to run ceviche.

## Import ceviche

Whenever you want to use a package in your Python script, you will have to import at. Before you can do this with `ceviche`, we need to add it to the system path, which is done by

```python
import sys
sys.path.append('path/to/ceviche-master')   # here: sys.path.append('../ceviche-master')
```

**Note:** If you actually installed ceviche with `pip install ceviche`, there are is no need for that and you may directly use `import ceviche` in your Python script.

## Running a Jupyter notebook

This repository comes along with a [Jupyter](https://jupyter.org/) notebook (`.ipynb` file format). You can open, edit and run this file with the program Jupyter Notebook, which you will already have installed if you choose Anaconda. Jupyter Notebook and its follow-up program JupyterLab are web-based user interfaces that run from your local computing environment. You may open them by running the command `jupyter notebook` or `jupyter lab` in your Anaconda Prompt or cmd. Redirect to the appropriate folder and open the file. In order to run the notebook cell-by-cell, you may use "Shift+Enter".

A Jupyter notebook consists of cells, which allows you to include not only code, but also text and plots. This comes in handy when you want to present your code. For pure programming, you may prefer a traditional IDE, such as [Spyder](https://www.spyder-ide.org/), where you run the code top-to-bottom.

## Working with Spyder

[Spyder](https://www.spyder-ide.org/) is an integrated development environment that comprises a text editor, a console, variable explorer and a debugger. Simply open Spyder, select a Python file (`.py` file format) and run the script by pressing the green play buttom or F5.

## Further reading

The [ceviche](https://github.com/fancompute/ceviche) package comes along with a [tutorial](https://github.com/fancompute/workshop-invdesign) covering examples, such as an inverse-designed waveguide mode converter and a wavelength multiplexer.

For additional reading on the concepts of optical inverse design, see these papers by [Shanhui Fan's group](https://web.stanford.edu/group/fan/).

 - T. W. Hughes, I. A. D. Williamson, M. Minkov, and S. Fan, "Forward-mode Differentiation of Maxwell’s Equations," ACS Photonics, Oct. 2019. [doi:10.1021/acsphotonics.9b01238](https://doi.org/10.1021/acsphotonics.9b01238)
 
  - T. W. Hughes*, M. Minkov*, I. A. D. Williamson, and S. Fan, "Adjoint Method and Inverse Design for Nonlinear Nanophotonic Devices," ACS Photonics, Dec. 2018. [doi:10.1021/acsphotonics.8b01522](https://doi.org/10.1021/acsphotonics.8b01522)
