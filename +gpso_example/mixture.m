function [out,dist,obj] = mixture( Ndim, Npeak, Neval, dist )
%
% [out,dist,obj] = gpso_example.mixture( Ndim, Npeak, Neval, dist )
%
% Optimise a Gaussian mixture using GPSO.
%
% INPUTS:
%
%   Ndim  Dimensionality of the problem
%  Npeak  Number of components in the mixture
%  Neval  Budget of objective evaluations
%   dist  If omitted, a mixture is generated automatically and returned in output.
%         If provided, this distribution is used for optimisation.
%         This allows to study the effects of various parameters while fixing the objective.
%
% OUTPUTS:
%
%    out  Solution found by GPSO.
%   dist  Mixture distribution used for this optimisation (can be passed as an input).
%    obj  GPSO instance used for optimisation.
%
% Examples:
%   [out,dist,obj] = gpso_example.mixture( 2, 10, 50 );
%   [out,dist,obj] = gpso_example.mixture( 6, 20, 500 );
%
% JH

    amp = 5; % max amplitude of individual component
    scl = 1; % deviation scale (max std)
    width = 10; % width of the actual hypercube (should not change anything)
    r_amp = 1/2; % amplitudes sampled between (r_amp,1)*amp
    r_scl = 1/2; % std sampled between (r_scl,1)*scl
    
    % create distributions
    if nargin < 4
        dist = cell(1,Npeak);
        for i = 1:Npeak
            mu = width * (0.1 + 0.8*rand(1,Ndim)); % in the inner 80% hypercube
            sigma = scl * (r_scl + (1-r_scl)*rand(1,Ndim)); % assume diagonal covariance (not isotropic)
            amp = amp * (r_amp + (1-r_amp)*rand(1));

            dist{i} = struct( ...
                'm', mu, 's', sigma, 'a', amp, ...
                'eval', @(x) amp*exp(-0.5*sum(bsxfun(@rdivide,bsxfun(@minus,x,mu),sigma).^2,2)), ...
                'sample', @(n) bsxfun(@plus,mu,bsxfun(@times,randn(n,Ndim),sigma)) ...
            );
        end
        dist = [dist{:}];
    end
    
    function y = objfun(x)
        y = zeros(size(x,1),1);
        for ii = 1:Npeak
            y = y + dist(ii).eval(x);
        end
    end

    domain = [zeros(Ndim,1),width*ones(Ndim,1)];
    obj = GPSO(); 
    out = obj.run( @objfun, domain, Neval, 'samp', Ndim^3 ); % use sampling in higher dimensions
    
    
    % summary results
    xpeak = vertcat(dist.m);
    ypeak = objfun(xpeak);
    
    [ybest,kbest] = max(ypeak);
    Dbest = norm( dist(kbest).m - out.sol.x );
    Rbest = (ybest-out.sol.f)/ybest;
    
    [Dclose,kclose] = min(sum(bsxfun(@minus,xpeak,out.sol.x).^2,2));
    Rclose = (ypeak(kclose)-out.sol.f)/ypeak(kclose);
    
    % NOTE:
    % These metrics are not exact; it's not trivial to find the global maximum of a mixture, 
    % so you might get negative regrets here.
    dk.print( '\n\nSolution found is %g units away from global optimum (relative regret: %.2f %%).', Dbest, 100*Rbest );
    dk.print( 'Closest local optimum is at %g units (relative regret: %.2f %%).', Dclose, 100*Rclose );

    figure; plotmatrix(out.samp.x);
    dk.ui.title('Scatter of evaluated samples across %d dimensions',Ndim);
    
end

% VISUALISATION IN 2D
%
% w = 10; % width of the hypercube
% p = 80; % grid size
% n = numel(dist);
% 
% [x,y] = meshgrid( linspace(0,w,p) );
% 
% G = [x(:),y(:)];
% z = zeros(p^2,1);
% for i = 1:n
%     z = z + dist(i).eval(G);
% end
% z = reshape(z,[p p]);
% 
% figure; colormap(gcf,'jet');
% surf(x,y,z,'FaceAlpha',0.9); hold on;
% plot3( out.samp.x(:,1), out.samp.x(:,2), out.samp.f, 'k*', 'MarkerSize', 8 );
% plot3( out.sol.x(:,1), out.sol.x(:,2), out.sol.f, 'rv', 'MarkerSize', 10 );
% hold off; axis vis3d;
% colorbar; xlabel('x'); ylabel('y');
