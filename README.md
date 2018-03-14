
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Gaussian-Process Surrogate Optimisation

> **NOTICE:**
> A new version of GPSO is being written, and this version will not be updated.
> A link to the new version will be posted here within a few months, but if you want to contribute, please contact me directly.

This implementation is adapted from [IMGPO](http://lis.csail.mit.edu/code/imgpo.html) and brings several improvements:

 - Correct update of GP hyperparameters, and subsequent update of upper-confidence bounds;
 - GP trained in normalised space, in accordance with the use of an isotropic covariance function;
 - Decoupling of GP surrogate, partition function and optimisation logics;
 - Correct count of GP- and non-GP-based samples;
 - Object orientation and events for improved clarity and interfacing;
 - Serialisation allowing the optimisation to be resumed at any stage;
 - Detailed configuration, and additional methods for post-analysis.

<img src="https://i.imgur.com/ATJZh57.gif" alt="Drawing" style="width: 50%; display: block; margin: 0 auto" />

## Installation

### Requirements

You will need to install [Deck](https://github.com/jhadida/deck); follow the instructions, and make sure it is on your Matlab path (e.g. call `dk.forward('All good')` to check).

You will also need to be able to compile Mex files with Matlab; type `mex -setup` from the console to make sure you're all set. If that doesn't work:
 - On **Linux**: just make sure you have `g++` and `gfortran` installed.
 - On **OSX**: this might be more complicated. The easiest is to install [Homebrew](https://brew.sh/), and then `brew install gcc`. If the compiling fails saying something about Xcode, then you will probably need to install the Command-Line Tools and Xcode, both of which can be found [here](https://developer.apple.com/download/more/) (if they are too big to download, check [this](https://apple.stackexchange.com/questions/252911/download-older-version-of-xcode) out).

### Install and compile

From the Matlab console:
```
folder = fullfile( userpath(), 'gpso' ); % or wherever you want to download it
setenv( 'GPSO_FOLDER', folder ); % to be used in system calls
!git clone https://github.com/jhadida/gpso.git "$GPSO_FOLDER"
addpath(folder);
gpml_start(); gpml_compile(); % compile GPML
gpso_example.peaks(); % opens a figure with a demo
```

If the compilation step fails, even though you have all the requirements, please [open an issue](https://github.com/jhadida/gpso/issues).

## Usage

### Running optimisation

Your objective function should be defined as a function accepting a single row-vector of parameters, and returning a scalar score to be **maximised**. For example, a 5-dimensional objective function should accept 1x5 vectors in input. 

Note that this algorithm is not suitable for combinatorial problems, or for problems with categorical inputs or outputs.
It also assumes that the search space is a cartesian domain, where an open interval of parameter-values (with finite bounds) is considered in each dimension.
The domain should be specified as an `Nd x 2` matrix where columns indicate respectively the lower and upper bounds in each dimension.

A "budget" of computation should be allocated for the optimisation. This budget is specified as the number of times the objective function can be evaluated (denoted `Neval`), the idea being that evaluations of the objective function dominate other operations in terms of runtime (in particular, the optimisation of GP hyperparameters at each iteration, which can take a few seconds). Note that this is _not_ a hard limit; the final number of evaluations may exceed the budget slightly, in order to complete the last iteration.

Here is a simple example with a 2-dimensional objective function:
```
objfun = @(x) exp(-norm(x./[1,2])); % ellipse aligned with y-axis
domain = [-1 1; -3 4]; % domain of different size in each dimension

obj = GPSO(); % create instance for optimisation
Neval = 50; % budget of 50 evaluations
output = obj.run( objfun, domain, Neval, 'tree', 3 ); % explore children leaf by growing partition tree 3 levels deep
```

There are several options that can be set for the optimisation (notably regarding the exploration method):
```
    function out = obj.run( objfun, domain, Neval, Xmet, Xprm, varargin )
    %
    % objfun:
    %   Function handle taking a candidate sample and returning a scalar.
    %   Expected input size should be 1 x Ndim.
    %   The optimisation MAXIMISES this function.
    %
    % domain:
    %   Ndim x 2 matrix specifying the boundaries of the cartesian domain.
    %
    % Neval:
    %   Maximum number of function evaluation.
    %   This can be considered as a "budget" for the optimisation.
    %   Note that the actual number of evaluations can exceed this value (usually not by much).
    %
    % Xmet: default 'tree'
    %   Method used for exploration of children intervals.
    %     - tree explores by recursively applying the partition function;
    %     - samp explores by randomly sampling the GP within the interval.
    %
    % Xprm: default 5 if xmet='tree', 5*Ndim^2 otherwise
    %   Parameter for the exploration step (depth if xmet='tree', number of samples otherwise).
    %   You can set the exploration method manually (attribute xmet), or via the configure method.
    %   Note that in dimension D, you need a depth at least D if you want each dimension to be 
    %   split at least once. This becomes rapidly impractical as D increases, so you might want
    %   to select the sampling method instead if D is large.
    %
    % KEY/VALUE OPTIONS:
    %
    % InitSample: by default, two vertices per dimension, equally spaced from the centre (diamond-shape).
    %   Initial set of points to use for initialisation.
    %   Input can be an array of coordinates, in which case points are evaluated before optimisation.
    %   Or a structure with fields {coord,score}, in which case they are used directly by the surrogate.
    %
    % UpdateCycle: by default, update at each iteration.
    %   Update GP hyperparameters every n cycles.
    %   See step_update for currently selected method.
    %
    % Verbose: default true
    %   Verbose switch.
```

In addition, the following options can be set using the method `.configure()` prior to running the optimisation:
```
    function self = obj.configure( varsigma, mu, sigma )
    %
    % varsigma: default erfcinv(0.01)
    %   Expected probability that UCB < f.
    %   Another way to understand this parameter is that it controls how 
    %   "optimistic" we are during the exploration step. At a point x 
    %   evaluated using GP, the UCB will be: mu(x) + varsigma*sigma(x).
    %
    % mu: default 0
    %   Initial value of constant mean function.
    %
    % sigma: default 1e-3
    %   Initial std of Gaussian likelihood function (in normalised units).
```

### Extras

Using events, serialising, exporting the partition tree. Will be documented shortly.

## Bugs

Please report anything fishy by creating a [new issue](https://github.com/jhadida/gpso/issues). 

## License

[GNU AGPL v3](https://tldrlegal.com/license/gnu-affero-general-public-license-v3-(agpl-3.0))
