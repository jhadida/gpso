function out = gpso_run( objfun, domain, neval, verb )
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
%    neval  Number of evaluations allowed, depends on the dimensionality of the problem and size of the domain.
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
        out = GPSO().run( objfun, domain, neval, 'Verbose', verb );
    else
        gpml_start;
        [x,fx,Xsamp,Fsamp] = imgpo_default( objfun, domain, neval, verb );
        gpml_stop;
        
        out.sol.x = x;
        out.sol.fx = fx;

        out.samp.x = Xsamp;
        out.samp.fx = Fsamp;
    end

end
