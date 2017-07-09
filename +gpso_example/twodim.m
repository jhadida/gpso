function [out,obj] = twodim( objfun, xdom, ydom, nmax, varargin )
%
% [out,obj] = gpso_example.twodim( objfun, xdom, ydom, nmax, varargin )
%
% JH

    opt = dk.obj.kwArgs(varargin{:});
    
    % drawing options
    drawopt.caxis = opt.get('caxis','auto');
    drawopt.eaxis = opt.get('eaxis','auto');
    drawopt.drawucb = opt.get('drawucb',false);
    
    % initialisation
    init = opt.get('init',[]);

    % exploration method
    Xmet = opt.get('xmet','tree');
    switch Xmet
        case 'tree'
            Xprm = opt.get('xprm',3);
        case 'samp'
            Xprm = opt.get('xprm',100);
        otherwise
            error('Unknown method.');
    end
    
    % create optimiser
    obj = GPSO();
    obj.addlistener( 'PostIteration', @callback );
    
    % generate reference surface
    nx = xdom(3);
    ny = ydom(3);
    gx = linspace( xdom(1), xdom(2), nx );
    gy = linspace( ydom(1), ydom(2), ny );
    [gx,gy] = meshgrid( gx, gy );
    
    grd = [gx(:),gy(:)]; % grid points as nx2 array
    ref = reshape(objfun(grd),size(gx)); % reference surface
    scl = [nx,ny]; % size of the grid
    
    % figures
    figure; colormap('jet'); dk.fig.resize(gcf,[500,1100]);
    
    % run optimisation
    domain = [ xdom(1:2); ydom(1:2) ];
    if isempty(init)
        out = obj.run( objfun, domain, nmax, Xmet, Xprm );
    else
        out = obj.run( objfun, domain, nmax, Xmet, Xprm, 'InitSample', init );
    end

    % callback function
    function callback( src, edata )
        
        tree = src.tree;
        srgt = src.srgt;
        
        [mu,sigma] = srgt.surrogate(grd);
        mu = reshape( mu, size(ref) );
        sigma = reshape( sigma, size(ref) );
        varsigma = src.srgt.get_varsigma();
        
        draw_surrogate( ref, mu, sigma, varsigma, drawopt );
        draw_tree( tree, scl );
        draw_samples(bsxfun( @times, srgt.samp_evaluated(false), scl )); 
        drawnow; pause(0.5);
        
    end
    
end

function draw_surrogate(r,m,s,v,opt)
    subplot(1,2,1);
    imagesc(r); caxis(opt.caxis); 
    set(gca,'YDir','normal');
    
        c=colorbar; 
        c.Label.String = 'Objective Function';
        title('Partition of Search Space');
    
    subplot(1,2,2);
    args = {'EdgeColor','none'};
    args = {};
    if opt.drawucb 
        surf(m+v*s,r-m,args{:}); 
    else
        surf(m,r-m,args{:});
    end
    caxis(opt.eaxis); axis vis3d; 
    
        c=colorbar; 
        c.Label.String = '(Objective - Surrogate)';
        title('Surrogate Function');
    
end

function draw_tree(T,s)
    subplot(1,2,1);
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
    legend([h1,h2],'Evaluated Points','Initial Points');
end
