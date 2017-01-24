function out = gpso_run( objfun, domain, niter, verb )
%
% out = gpso_run( objfun, domain, niter, verb=true )
%
% Run GP surrogate optimisation with:
%
%   objfun  Objective function to be MAXIMISED.
%           Should take an 1xD input vector of parameters, where D is the dimensionality of the optimisation problem,
%           and return a scalar score to be maximised.
%           The function can internally save the sample, along with any useful analytic (score, etc.) if needed.
%
%   domain  Cartesian domain Dx2 for parameter sampling. 
%           Inputs to the objective function are guaranteed to be in this domain.
%
%    niter  Number of iterations to run, depends on the dimensionality of the problem and size of the domain.
%    sigma  Initial std used by the likelihood function.
%           Note that the domain is internally rescaled to [0,1]^D, so the std value should be normalised.
%
%     verb  (optional) Verbosity flag, true by default.
%
% The output is a structure with fields:
%
%   sol.x    Optimal parameters (given the number of iterations).
%   sol.fx   Value of objective function at optimum.
%
%   samp.x   All candidate parameters sampled during optimisation.
%   samp.fx  Corresponding values of objective function.
%
% JH

    USE_NEW=true;

    if nargin < 6, verb=true; end
    assert( size(domain,2)==2, 'Domain should be Nd x 2.' );
    
    if USE_NEW
        obj = GPSO().set_defaults(); 
        out = obj.run( objfun, domain, niter, verb );
    else
        gpml_start;
        [x,fx,Xsamp,Fsamp] = imgpo_default( objfun, domain, niter, verb );
        gpml_stop;
        
        out.sol.x = x;
        out.sol.fx = fx;

        out.samp.x = Xsamp;
        out.samp.fx = Fsamp;
    end

end
