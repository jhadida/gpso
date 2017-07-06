
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Gaussian-Process Surrogate Optimisation

This is a new implementation of [IMGPO](http://lis.csail.mit.edu/code/imgpo.html) which brings several improvements:

 - Correct update of GP hyperparameters, and subsequent update of UCBs;
 - GP trained in normalised space, to honour isotropic covariance assumptions;
 - Correct count of GP- and non-GP-based samples;
 - Decoupling of GP surrogate, partition tree and optimisation logic;
 - Object orientation and events for better clarity and interfacing;
 - Serialisation, configuration and numerous methods for analysis.

## Installation

You will need [Deck](https://github.com/sheljohn/deck) to use this toolbox.
If it is not already installed, navigate to the folder where you want to put it (eg `/Users/me/Documents/MATLAB`), and type:
```
git clone https://github.com/sheljohn/deck.git
```
This will create a folder `deck/`; add it to your path, and type `dk_startup` from the Matlab console.
If you need to use this toolbox frequently, add these last commands to your [`startup.m`](https://uk.mathworks.com/help/matlab/ref/startup.html).

You will also need to be able to compile Mex files; make sure Matlab is set up properly (if you are on OSX, this involves installing the Command-Line Tools and Xcode, you might also want to use Homebrew to install `gcc` and/or up-to-date versions of `clang`). For a quick verification you can type `mex -setup` from the console.

Once you have Deck installed, and that Mex is setup up properly, install GPSO if your folder of chocie with:
```
git clone https://github.com/sheljohn/gpso.git
```
This will create a folder `gpso/`; add it to your path, and type `gpml_compile` from the Matlab console.

## Usage

### Running optimisation

Create, configure, run, resume

### Extras

Using events, serialising, exporting the tree

## Examples

Can be accessed through `gpso_example.*`

## License

[GNU AGPL v3](https://tldrlegal.com/license/gnu-affero-general-public-license-v3-(agpl-3.0))
