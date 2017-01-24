function example_peaks( neval, xdom, ydom )

    if nargin < 2, xdom=[-3 3 100]; end
    if nargin < 3, ydom=xdom; end

    % optimiser and listener
    obj = GPSO().set_defaults();
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
    obj.run( @objfun, domain, neval );

    % callback function
    function callback( src, edata )
        sur = src.gp_call( src.normalise(grd) );
        sur = reshape( sur, size(ref) );
        
        ns = src.Nsamp;
        td = src.Tdepth;
        
        draw_surrogate( sur, ref-sur );
        draw_tree( src.tree, td, scl );
        draw_samples(bsxfun( @times, src.samp.x(1:ns,:), scl ));
        pause;
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

function draw_surrogate(z,d)
    surf(10+z); 
    hold on;
    imagesc(d); 
    colormap('jet'); caxis([-8 8]); colorbar; 
    hold off;
end

function draw_tree(T,d,s)
    hold on;
    for h = 1:d
        L = find(T(h).leaf);
        n = numel(L);
        for k = 1:n
            draw_rectangle(s(1)*T(h).x_min(L(k),:),s(2)*T(h).x_max(L(k),:));
        end
    end
    hold off;
end

function draw_rectangle(xmin,xmax)
    x = [xmin(1),xmin(1),xmax(1),xmax(1),xmin(1)];
    y = [xmin(2),xmax(2),xmax(2),xmin(2),xmin(2)];
    z = ones(1,5);
    plot3(x,y,z,'k-');
end

function draw_samples(X)
    hold on; n = size(X,1);
    plot3(X(:,1),X(:,2),ones(n,1),'r*');
    hold off;
end
