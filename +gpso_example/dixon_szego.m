function [out,obj] = dixon_szego( nmax, xdom, ydom, varargin )
%
% [out,obj] = gpso_example.dixon_szego( nmax, xdom, ydom, varargin )
%
% Example:
%
%   [out,obj] = gpso_example.dixon_szego( 100, [-2 2 80] );
%
% JH

    if nargin < 1, nmax = 50; end
    if nargin < 2, xdom=[-2 2 80]; end
    if nargin < 3, ydom=xdom; end

    args = [ {'caxis',[0 11],'eaxis',[-10 10]}, varargin ];
    [out,obj] = gpso_example.twodim( @objfun, xdom, ydom, nmax, args{:} );
    
end

function z = objfun(in)

    x = in(:,1);
    y = in(:,2);
    z = (4-2.1*x.^2+ x.^4/3).*x.^2 + x.*y + 4*(y.^2-1).*y.^2;
    z = 10 - z;

end
