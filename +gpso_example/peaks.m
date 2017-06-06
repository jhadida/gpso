function [out,obj,foo] = peaks( nmax, xdom, ydom, varargin )
%
% [out,obj] = gpso_example.peaks( nmax, xdom, ydom, varargin )
%
% Examples: 
%   gpso_example.peaks(50);
%   gpso_example.peaks(50,[-9 7 80],[-10 13 80]);
%   gpso_example.peaks(50,[-9 7 80],[-2 13 80]);
%   gpso_example.peaks(80,[-5 10 80],[-5 17 80]);
%
% JH

    if nargin < 1, nmax = 30; end
    if nargin < 2, xdom=[-3 3 80]; end
    if nargin < 3, ydom=xdom; end

    args = [ {'caxis',[-8 8],'eaxis',[-8 8]}, varargin ]; 
    [out,obj] = gpso_example.twodim( @objfun, xdom, ydom, nmax, args{:} );
    foo = @objfun;
    
end

function z = objfun(in)

    DO_ROTATE=true;
    OFFSET=0;

    x = in(:,1);
    y = in(:,2);
    
    if DO_ROTATE
        ct = cos(pi/4);
        st = sin(pi/4);

        xn = ct*x + st*y;
        yn = ct*y - st*x;

        x = xn;
        y = yn;
    end
    
    z =  3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) ...
        - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) ...
        - 1/3*exp(-(x+1).^2 - y.^2) - OFFSET;

end
