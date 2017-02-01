function obj = example_peaks( nmax, xdom, ydom )
%
% Example work: 
% example_peaks(50);
% example_peaks(80,[-9 7 50],[-10 13 50]);
% example_peaks(80,[-9 7 50],[-2 13 50]);
%
% Example fail: 
% example_peaks(100,[-5 10 100],[-5 17 100]);
%
% JH

    if nargin < 2, xdom=[-3 3 100]; end
    if nargin < 3, ydom=xdom; end

    % optimiser and listener
    obj = GPSO().configure();
    obj.addlistener( 'PostIteration', @callback );
    
    % generate reference surface
    nx = xdom(3);
    ny = ydom(3);
    gx = linspace( xdom(1), xdom(2), nx );
    gy = linspace( ydom(1), ydom(2), ny );
    [gx,gy] = meshgrid( gx, gy );
    
    ref = objfun( gx, gy );
    grd = [gx(:),gy(:)];
    scl = [nx,ny];
    
    % run optimisation
    domain = [ xdom(1:2); ydom(1:2) ];
    obj.run( @objfun, domain, nmax );

    % callback function
    function callback( src, edata )
        
        tree = src.tree;
        srgt = src.srgt;
        
        [mu,sigma] = srgt.surrogate(grd);
        sur = reshape( mu + sigma, size(ref) );
        
        draw_surrogate( ref, sur );
        draw_tree( tree, scl );
        draw_samples(bsxfun( @times, srgt.samp_evaluated, scl ));
        pause(0.5);
        %dk.ui.fig.print( gcf, 'example_fail/iter_%02d', src.Niter );
    end
    
end

function z = objfun(x,y)

    if nargin == 1
        y = x(2);
        x = x(1);
    end
    z =  3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) ...
        - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) ...
        - 1/3*exp(-(x+1).^2 - y.^2);

end

function draw_surrogate(r,s)
    surf(s-15,r-s); 
    hold on;
    imagesc(r); 
    colormap('jet'); 
    caxis([-8 8]); colorbar; 
    axis tight; hold off;
end

function draw_tree(T,s)
    hold on; d=T.depth;
    for h = 1:d
        L = find(T.level(h).leaf);
        n = numel(L);
        for k = 1:n
            draw_rectangle(...
                s(1) * T.lower(h,L(k)), ...
                s(2) * T.upper(h,L(k)) ...
            );
        end
    end
    hold off;
end

function draw_rectangle(xmin,xmax)
    x = [xmin(1),xmin(1),xmax(1),xmax(1),xmin(1)];
    y = [xmin(2),xmax(2),xmax(2),xmin(2),xmin(2)];
    z = 0.1*ones(1,5);
    plot3(x,y,z,'k-');
end

function draw_samples(X)
    hold on; n = size(X,1);
    plot3(X(:,1),X(:,2),0.1*ones(n,1),'r*');
    hold off;
end
