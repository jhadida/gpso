classdef GPSO_Tree < handle
%
% Free software provided under AGPLv3 license (see README).
% Copyright Jonathan Hadida (jhadida@fmrib.ox.ac.uk), July 2017.

    properties (SetAccess = private)
        level % struct-array storing tree contents level by level
        Nl,Ns % number of leaves/splits
    end
    
    properties (Transient,Dependent)
        depth
    end
    
    methods
        
        function self = GPSO_Tree()
            self.clear();
        end
        
        function self=clear(self)
            self.level = [];
            self.Nl = 0;
            self.Ns = 0;
        end
        
        % serialise data to be saved
        function D = serialise(self)
            F = {'level','Nl','Ns'};
            n = numel(F);
            D = struct();
            
            for i = 1:n
                f = F{i};
                D.(f) = self.(f);
            end
            D.version = '0.2';
        end
        function self=unserialise(self,D)
            F = {'level','Nl','Ns'};
            n = numel(F);
            
            for i = 1:n
                f = F{i};
                self.(f) = D.(f);
            end
            
            switch D.version
                case '0.1'
                    % used to store level and index, but level is useless since it's always the one just above
                    for h = 1:self.depth
                        assert( all( self.level(h).parent(:,1) == h-1 ), 'Bad ancestry record.' );
                        self.level(h).parent = self.level(h).parent(:,2)';
                    end
                    if self.depth
                        self.level(1).parent = 0; % root index is 0
                    end
            end
        end
        
        function self=init(self,ndim,rid)
        %
        % ndim: dimensionality of search space
        % rid: storage index of tree root (centre of hyperdomain)
        % 
            
            if nargin < 3, rid=1; end
            
            % initialise tree
            T.parent = 0;
            T.lower  = zeros(1,ndim);
            T.upper  = ones(1,ndim);
            T.samp   = rid;
            T.leaf   = true;
            
            self.level = T;
            self.Nl = 1;
            self.Ns = 0;
            
        end
        
        function d = get.depth(self)
            d = numel(self.level);
        end
        function w = width(self,h)
            w = numel(self.level(h).samp);
        end
        
        % quick access
        function p = parent(self,h,k)
            p = self.level(h).parent(k);
        end
        function l = lower(self,h,k)
            l = self.level(h).lower(k,:);
        end
        function u = upper(self,h,k)
            u = self.level(h).upper(k,:);
        end
        function s = samp(self,h,k)
            s = self.level(h).samp(k);
        end
        function l = leaf(self,h,k)
            l = self.level(h).leaf(k);
        end
        
        % works with k vector, but h should be scalar
        % not super efficient, don't use too often
        function n = node(self,h,k)
            n.parent = self.parent(h,k);
            n.lower  = self.lower(h,k);
            n.upper  = self.upper(h,k);
            n.samp   = self.samp(h,k);
            n.leaf   = self.leaf(h,k);
        end
 
        function child=split(self,h,k,srgt,xmet,xprm)
        % 
        % h,k: level+id of node to split
        % srgt: surrogate object
        % xmet,xprm: exploration options
        %
        
            % make sure it's a leaf
            assert( self.leaf(h,k), '[bug] Splitting non-leaf node.' );
            
            % select exploration method
            switch lower(xmet)
                case {'tree','grow'}
                    vcat = @(x) vertcat(x.coord);
                    xfun = @(node) vcat(self.grow(node,xprm));
                case {'samp','urand','sample'}
                    xfun = @(node) self.sample(node,xprm);
                otherwise
                    error( 'Unknown exploration method %s', xmet );
            end
            
            % children are in the next level
            m = h+1;
            if m > self.depth % initialise if necessary
                self.level(m).leaf = true(0); 
            end
            
            % split node along largest dimension
            parent = self.node(h,k);
            child  = recursive_split( parent );
            
            % evaluate each new leaf
            nc = 3; % == numel(child)
            varsigma = srgt.get_varsigma();
            [~,child(1).best] = srgt.gp_eval( xfun(child(1)), varsigma );
            [~,child(2).best] = srgt.gp_eval( xfun(child(2)), varsigma );
            [~,child(3).best] = srgt.gp_eval( xfun(child(3)), varsigma );
            
            % insert children into surrogate
            sid = srgt.append( vertcat(child.coord), vertcat(child.best), true );
            
            % insert children into tree
            self.level(m).parent = [self.level(m).parent, k*ones(1,nc)];
            self.level(m).lower  = [self.level(m).lower; vertcat(child.lower)];
            self.level(m).upper  = [self.level(m).upper; vertcat(child.upper)];
            self.level(m).samp   = [self.level(m).samp, sid];
            self.level(m).leaf   = [self.level(m).leaf, true(1,nc)];
            
            % parent is no longer a leaf
            self.level(h).leaf(k) = false;
            
            % update split+leaf counts
            self.Ns = self.Ns+1;
            self.Nl = self.Nl+nc-1;
            
        end

        function children = grow(self,node,d)
        %
        % node: node structure, or [h,k] vector
        % d: depth of the subtree to grow
        %   (WARNING: tree grows exponentially fast!)
        %
        % Grow tree from node (h,k) without saving anything.
        % Return a struct-array of children nodes with fields {lower,upper,coord}.
        %
        % NOTE: coordinates are NORMALISED here
        %
        
            dk.assert( d <= 8, [ ... 
                'This is safeguard error to prevent deep tree explorations.\n' ...
                'If you meant to set the option xmet="tree" with a depth of %d (%d samples),\n' ...
                'then please comment this message in the method grow.\n' ...
            ], d, 3^d );
        
            
            if ~isstruct(node)
                node = self.node( node(1), node(2) );
            end
            node.coord = (node.lower + node.upper)/2;
            children = recursive_split( node, d );
            
        end
        
        function points = sample(self,node,ns)
        %
        % node: node structure, or [h,k] vector
        % ns: number of random points to sample
        %
        % Sample n points within node (h,k) uniformly randomly.
        %
        % NOTE: coordinates are NORMALISED here
        %
        
            if ~isstruct(node)
                node = self.node( node(1), node(2) );
            end
            
            nd = numel(node.upper);
            delta = node.upper - node.lower;
            points = bsxfun( @times, rand(ns,nd), delta );
            points = bsxfun( @plus, points, node.lower );
            
        end
        
        function T = export_compact(self)
        %
        % Export tree as a structure with fields:
        %
        %     parent  Index of the parent node (in that array, not in this object).
        %   children  Zero for leaf-nodes, otherwise index of first child.
        %     sample  Index of the corresponding surrogate sample.
        %      depth  Depth of the node in the tree.
        %      order  Index of youth (higher values mean younger).
        %
        % Note: children are necessarily next to each other, so if C is the index of the first child,
        %   then the other children are C+1 and C+2.
        % 
        % JH
        
            d = self.depth; % tree depth
            w = arrayfun( @(x) numel(x.samp), self.level ); % width of each level
            n = sum(w); % total number of nodes
            
            % allocate output
            T = zeros(1,n);
            T = struct( 'n', n, 'd', d, ...
                'parent', T, 'children', T, 'sample', T, 'depth', T, 'order', T ...
            );
            
            % first pass:
            %   set depth, sample indices, and re-index parents
            b = 1;
            e = 0;
            for h = 1:d
                o = b-1;
                b = e+1;
                e = b+w(h)-1;
                
                T.parent(b:e) = o + self.level(h).parent;
                T.sample(b:e) = self.level(h).samp;
                T.depth(b:e) = h;
            end
            
            % second pass: set children 
            for i = 1:n
                p = T.parent(i); 
                if p>0 && T.children(p)==0
                    T.children(p) = i;
                end
            end
            
            % third pass: set order
            for i = 2:3:n
                k = i + [0 1 2];
                T.order(k) = max(T.sample(k));
            end
            T.order(1) = T.sample(1); % set the root
            
            % re-index the order
            u = unique(T.order);
            r(u) = 1:numel(u);
            T.order = r(T.order);
            
        end
        
        function T = export_dkTree(self)
        %
        % Export as dk.ds.Tree instance.
            
            C = self.export_compact();
            C.index = ones(1,C.n);
            
            T = dk.ds.Tree( struct('sid', C.sample(1), 'order', C.order(1)) );
            for d = 2:C.d
                
                k = find(C.depth == d);
                n = numel(k);
                
                for i = 1:n
                    ki = k(i);
                    pi = C.parent(ki);
                    C.index(ki) = T.add_node( C.index(pi), 'sid', C.sample(ki), 'order', C.order(ki) );
                end
                
            end
            
        end
        
    end
    
end


% 
%     node.lower                 node.upper
% Lvl      \                         /
% h:        =---------node----------=
% 
%
% h+1:      =---L---=---M---=---R---=
%          /        |       |        \
%        Pmin     Lmax     Rmin     Pmax
%
function children = recursive_split(node,count)

    if nargin < 2, count=1; end

    % bounds of parent node
    Pmin = node.lower;
    Pmax = node.upper;
    
    % halting condition
    if count == 0 
        children = node;
        return; 
    end
    
    % barycenter of children subintervals
    M = (Pmin + Pmax) / 2;
    L = M;
    R = M;
    
    [~,s] = max( Pmax - Pmin ); % split along largest dimension
    L(s)  = (5*Pmin(s) +   Pmax(s))/6;
    R(s)  = (  Pmin(s) + 5*Pmax(s))/6;
    
    % compute bounds of children
    Lmax = Pmax;
    Rmin = Pmin;
    Mmin = Pmin;
    Mmax = Pmax;
    
    Lmax(s) = (2*Pmin(s) +   Pmax(s))/3.0;
    Rmin(s) = (  Pmin(s) + 2*Pmax(s))/3.0;
    Mmin(s) = Lmax(s);
    Mmax(s) = Rmin(s);
    
    % pack as struct-array
    make_node = @(ll,uu,xx) struct('lower',ll,'upper',uu,'coord',xx,'sdim',s);
    children  = [ ... WARNING: Order matters!
        recursive_split(make_node(Pmin,Lmax,L), count-1), ... left
        recursive_split(make_node(Rmin,Pmax,R), count-1), ... right
        recursive_split(make_node(Mmin,Mmax,M), count-1)  ... middle
    ];
    
end
