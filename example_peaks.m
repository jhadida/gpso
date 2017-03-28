function [out,obj] = example_peaks( nmax, xdom, ydom, init )
%
% [out,obj] = example_peaks( nmax, xdom, ydom )
%
% Examples: 
%   example_peaks(50);
%   example_peaks(50,[-9 7 80],[-10 13 80]);
%   example_peaks(50,[-9 7 80],[-2 13 80]);
%   example_peaks(80,[-5 10 80],[-5 17 80]);
%
% JH

    global PAPER_MODE;
    PAPER_MODE=false;

    if nargin < 2, xdom=[-3 3 80]; end
    if nargin < 3, ydom=xdom; end

    % optimiser and listener
    obj = GPSO('tree'); XSIZE=3;
    %obj = GPSO('samp'); XSIZE=100;
    obj.addlistener( 'PostIteration', @callback );
    
    % generate reference surface
    nx = xdom(3);
    ny = ydom(3);
    gx = linspace( xdom(1), xdom(2), nx );
    gy = linspace( ydom(1), ydom(2), ny );
    [gx,gy] = meshgrid( gx, gy );
    
    ref = objfun( gx, gy ); % reference surface
    grd = [gx(:),gy(:)]; % grid points as nx2 array
    scl = [nx,ny]; % size of the grid
    
    % figures
    if PAPER_MODE
        figure(1); clf; colormap('jet'); dk.ui.fig.resize(gcf,[700,800]);
        figure(2); clf; colormap('jet'); dk.ui.fig.resize(gcf,[700,800]);
    else
        figure; colormap('jet'); dk.ui.fig.resize(gcf,[500,1100]);
    end
    
    % run optimisation
    domain = [ xdom(1:2); ydom(1:2) ];
    if nargin > 3
        out = obj.run( @objfun, domain, nmax, 'ExploreSize', XSIZE, 'InitSample', init );
    else
        out = obj.run( @objfun, domain, nmax, 'ExploreSize', XSIZE );
    end

    % callback function
    function callback( src, edata )
        
        tree = src.tree;
        srgt = src.srgt;
        
        [mu,sigma] = srgt.surrogate(grd);
        mu = reshape( mu, size(ref) );
        sigma = reshape( sigma, size(ref) );
        varsigma = src.srgt.get_varsigma();
        
        draw_surrogate( ref, mu, sigma, varsigma );
        draw_tree( tree, scl );
        draw_samples(bsxfun( @times, srgt.samp_evaluated(false), scl )); 
        drawnow; pause(0.5);
        
        if PAPER_MODE
            exportfig( [1 2], '/Users/jhadida/Desktop/IMG/gpso/paper_v4', sprintf('iter_%02d_f%%d.png',src.Niter) );
        end
    end

    % show axis on separate figure
    if PAPER_MODE
        figure; colormap('jet'); n = 1024;
        imagesc((0:n)'); grid off; box off;
        ytl = dk.arrayfun( @num2str, -8:2:8, false );
        set(gca,'xtick',[],'ytick',linspace(5,n-5,numel(ytl)),'yticklabel',ytl,'yaxislocation','right','ydir','normal');
        ylabel( 'Objective Function' ); 
        dk.ui.fig.resize( gcf, [800 100] );
    end
    
end

function z = objfun(x,y)

    DO_ROTATE=true;
    OFFSET=0;

    if nargin == 1
        y = x(2);
        x = x(1);
    end
    
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

function draw_surrogate(r,m,s,v)
    
    global PAPER_MODE;
    
    if PAPER_MODE
        figure(1); 
    else
        subplot(1,2,1);
    end
    imagesc(r); caxis([-8 8]); 
    set(gca,'YDir','normal');
    if ~PAPER_MODE
        c=colorbar; 
        c.Label.String = 'Objective Function';
        title('Partition of Search Space');
    end
    
    if PAPER_MODE
        figure(2); 
    else
        subplot(1,2,2);
    end
    surf(m+v*s,r-m); caxis([-8 8]); axis vis3d; 
    if ~PAPER_MODE
        c=colorbar; 
        c.Label.String = '(Objective - Surrogate)';
        title('Surrogate Function');
    end
    
end

function draw_tree(T,s)
    global PAPER_MODE;
    if PAPER_MODE
        figure(1); 
    else
        subplot(1,2,1);
    end
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
    h2 = plot( X(1:4,1), X(1:4,2), 'kp', 'MarkerSize', 8 );
    h1 = plot( X(5:end,1), X(5:end,2), 'k*', 'MarkerSize', 8 );
    hold off;
    legend([h1,h2],'Evaluated Points','L1-ball vertices');
end

function exportfig( fig_id, folder, nametpl )
    
    for f = fig_id
        dk.ui.fig.export(figure(f),fullfile(folder,sprintf(nametpl,f)),'paper');
    end

end
