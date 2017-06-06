function [out,obj] = rosenbrock( nmax, xdom, ydom, varargin )
%
% [out,obj] = gpso_example.rosenbrock( nmax, xdom, ydom, varargin )
%
% Example:
%
%   [out,obj] = gpso_example.rosenbrock( 100, [-1 1 80] );
%
% JH

    if nargin < 1, nmax = 100; end
    if nargin < 2, xdom=[-1 1 80]; end
    if nargin < 3, ydom=xdom; end

    args = [ {'caxis',[-1 1],'eaxis',[-1 1]}, varargin ];
    [out,obj] = gpso_example.twodim( @objfun, xdom, ydom, nmax, args{:} );
    
end

function z = objfun(in)

    x = in(:,1);
    y = in(:,2);
    z = -exp(x-2*x.^2-y.^2).*sin(6*(x+y+x.*y.^2));

end
