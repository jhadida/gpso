
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Gaussian-Process Surrogate Optimisation

This is a new implementation (from scratch) of [IMGPO](http://lis.csail.mit.edu/code/imgpo.html) which brings several improvements:

 - Correct update of GP hyperparameters, and subsequent update of UCBs;
 - GP trained in normalised space, to honour isotropic covariance assumptions;
 - Decoupling of GP surrogate, partition tree and optimisation logic;
 - Correct count of GP- and non-GP-based samples;
 - Object orientation and events for better clarity and interfacing;
 - Serialisation, configuration and numerous methods for analysis.

## Installation

You will need [Deck](https://github.com/sheljohn/deck) to use this toolbox.
If it is not already installed, navigate to the folder where you want to put it (eg `~/Documents/MATLAB`), and type:
```
git clone https://github.com/sheljohn/deck.git
```
This will create a folder `deck/`.
Add this folder to your path, and type `dk_startup` from the Matlab console.
If you need to use this toolbox frequently, add these last commands to your [`startup.m`](https://uk.mathworks.com/help/matlab/ref/startup.html).

You will also need to be able to compile Mex files; make sure Matlab is set up properly (if you are on OSX, this involves installing the Command-Line Tools and Xcode, you might also want to use Homebrew to install `gcc` and/or up-to-date versions of `clang`). For a quick verification you can type `mex -setup` from the console.

Once you have Deck installed, and that Mex is setup up properly, install GPSO in your folder of choice with:
```
git clone https://github.com/sheljohn/gpso.git
```
This will create a folder `gpso/`. 
Add this folder to your path, and type `gpml_compile` from the Matlab console.

## Usage

### Running optimisation

Your objective function should be defined as a function handle accepting a single row-vector of candidate parameters, and returning a scalar score to be **maximised**. For example, a 5-dimensional objective function should accept 1x5 vectors in input. 

Note that this algorithm is not suitable for combinatorial problems, or for problems with categorical inputs or outputs.
It also assumes that the search space is a cartesian domain, where a closed interval of parameter values is considered in each dimension.
The domain should be specified as an `Nd x 2` matrix where columns indicate respectively the lower and upper bounds in each dimension.

A "budget" of computation should be allocated for the optimisation. This budget is specified as the number of times the objective function can be evaluated (denoted `Neval`), the idea being that evaluations of the objective function dominate other operations (in particular the optimisation of GP hyperparameters at each iteration). Note that ths is _not_ a hard limit; the final number of evaluations may exceed the budget slightly, in order to complete the last iteration.

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
    % InitSample: default L1-ball vertices
    %   Initial set of points to use for initialisation.
    %   Input can be an array of coordinates, in which case points are evaluated before optimisation.
    %   Or a structure with fields {coord,score}, in which case they are used directly by the surrogate.
    %
    % UpdateCycle: default 1
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

Using events, serialising, exporting the tree

## Examples

Can be accessed through `gpso_example.*`

## License

[GNU AGPL v3](https://tldrlegal.com/license/gnu-affero-general-public-license-v3-(agpl-3.0))
