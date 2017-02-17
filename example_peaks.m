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
    
    itc = -1;
    ref = objfun( gx, gy );
    grd = [gx(:),gy(:)];
    scl = [nx,ny];
    
    % run optimisation
    figure(1); clf; colormap('jet'); dk.ui.fig.resize(gcf,[700,800]);
    figure(2); clf; colormap('jet'); dk.ui.fig.resize(gcf,[700,800]);
    %figure; dk.ui.fig.resize(gcf,[500,1100]);
    domain = [ xdom(1:2); ydom(1:2) ];
    obj.run( @objfun, domain, nmax, 30 );

    % callback function
    function callback( src, edata )
        
        tree = src.tree;
        srgt = src.srgt;
        
        [mu,sigma] = srgt.surrogate(grd);
        mu = reshape( mu, size(ref) );
        sigma = reshape( sigma, size(ref) );
        
        draw_surrogate( ref, mu, sigma );
        draw_tree( tree, scl );
        draw_samples(bsxfun( @times, srgt.samp_evaluated, scl )); drawnow; pause(0.5);
        
        itc = itc+1;
        %exportfig( [1 2], '/Users/jhadida/Desktop/IMG/gpso', sprintf('iter_%02d_f%%d.png',itc) );
    end
    
end

function z = objfun(x,y)

    if nargin == 1
        y = x(2);
        x = x(1);
    end
    
    % rotation
    ct = cos(pi/4);
    st = sin(pi/4);
    
    xn = ct*x + st*y;
    yn = ct*y - st*x;
    
    x = xn;
    y = yn;
    z =  3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) ...
        - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) ...
        - 1/3*exp(-(x+1).^2 - y.^2);

end

function draw_surrogate(r,m,s)
    
    figure(1); %subplot(1,2,1); 
    imagesc(r); caxis([-8 8]); c=colorbar; 
    c.Label.String = 'Objective Function';
    set(gca,'YDir','normal');
    title('Partition of Search Space');
    
    figure(2); %subplot(1,2,2)
    surf(m+s,r-m); 
    caxis([-8 8]); c=colorbar; 
    c.Label.String = '(Objective - Surrogate)';
    axis vis3d; title('Surrogate Function');
    
end

function draw_tree(T,s)
    figure(1); %subplot(1,2,1);
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
    plot(x,y,'k-');
end

function draw_samples(X)
    hold on; 
    h2 = plot( X(1:4,1), X(1:4,2), 'kp' );
    h1 = plot( X(5:end,1), X(5:end,2), 'k*', 'MarkerSize', 8 );
    hold off;
    legend([h1,h2],'Evaluated Points','L1-ball vertices');
end

function exportfig( fig_id, folder, nametpl )
    
    for f = fig_id
        dk.ui.fig.export(figure(f),fullfile(folder,sprintf(nametpl,f)),'normal');
    end

end
