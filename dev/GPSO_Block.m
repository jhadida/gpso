classdef GPSO_Block < handle
%
% Example:
%
% A = GPSO_Block( 5, 3 );
% A.setr( 3, [1 2 3] )
% A.setr( 7, [2 4 6] )
% m = A.compress(); disp(m'); disp(A);
% 
% A.setr(6,[1 0 1]); A.delr(6)
% m = A.compress(); disp(m'); disp(A);
%
    
    properties (SetAccess = private)
        block;
        bsize;
        nrows, ncols;
    end
    
    properties (Dependent)
        nblocks, capacity;
    end
    
    methods
        
        function self = GPSO_Block( varargin )
            self.clear();
            if nargin > 0
                self.create(varargin{:});
            end
        end
        
        function n = get.nblocks(self)
            n = numel(self.block);
        end
        function n = get.capacity(self)
            n = self.nblocks * self.bsize;
        end
        
        function self=clear(self)
            self.block = struct('x',[],'u',[]);
            self.nrows = 0;
            self.ncols = 0;
            self.bsize = 1;
        end
        
        function self=create(self,nrows,ncols)
            self.block = struct('x',[],'u',[]);
            self.block.x = nan(nrows,ncols);
            self.block.u = false(nrows,1);
            
            self.nrows = 0;
            self.ncols = ncols;
            self.bsize = nrows;
        end
        
        % assign elements
        function self=setr(self,r,val)
            [b,k] = self.ind2blk(r);
            while b > self.nblocks
                self.blockinit(self.nblocks+1);
            end
                        
            self.block(b).x(k,:) = val;
            self.block(b).u(k) = true;
            self.nrows = max( self.nrows, r );
        end
        function r=append(self,val) % returns index
            r = self.nrows+1;
            self.setr(r,val);
        end
        
        % delete/get single element
        function self=delr(self,r)
            assert( isscalar(r), 'Index should be scalar (use delm instead).' );
            [b,k] = self.ind2blk(r);
            assert( b <= self.nblocks, 'Index out of bounds.' );
            self.block(b).u(k) = false;
        end
        function x=getr(self,r)
            [b,k] = self.ind2blk(r);
            assert( b <= self.nblocks, 'Index out of bounds.' );
            x = self.block(b).x(k,:);
        end
        
        % delete/get multiple elements
        function self=delm(self,r)
            for i = 1:numel(r)
                self.delr(r(i));
            end
        end
        function x=getm(self,r)
            [b,k] = self.ind2blk(r(:));
            u = unique(b);
            n = numel(u);
            x = cell(1,n);
            
            assert( all(u <= self.nblocks), 'Index out of bounds.' );
            for i = 1:n
                x{i} = self.block(u(i)).x(k(b == u(i)),:);
            end
            x = vertcat(x{:});
        end
        
        function remap=compress(self)
            
            % concatenate use vectors
            used  = vertcat(self.block.u);
            remap = cumsum(used) .* used;
            
            % compress blocks
            m = max(remap);
            b = 0;
            n = 0;
            
            while m-n >= self.bsize
                b = b + 1;
                p = n + 1;
                n = n + self.bsize;
                
                r = find( (remap >= p) & (remap <= n) );
                self.block(b).x = self.getm(r); %#ok
                self.block(b).u = true(self.bsize,1);
            end
            
            % last ones
            if n < m
                b = b + 1;
                r = find( remap > n );
                n = numel(r);
                
                x = nan(self.bsize,self.ncols);
                u = false(self.bsize,1);
                
                x(1:n,:) = self.getm(r);
                u(1:n)   = true;
                
                self.block(b).x = x;
                self.block(b).u = u;
            end
            self.block = self.block(1:b);
            
        end
        
        function disp(self)
            for b = 1:self.nblocks
                fprintf('\nBlock %d:\n',b);
                disp([self.block(b).u, self.block(b).x]);
            end
        end
        
    end
    
    methods (Hidden,Access=private)
        
        function [b,k] = ind2blk(self,i)
            b = 1 + fix( (i-1) / self.bsize );
            k = i - self.blk2ind(b,0);
        end
        function i = blk2ind(self,b,k)
            i = (b-1)*self.bsize + k;
        end
        
        function blockinit(self,b)
            self.block(b).x = nan( self.bsize, self.ncols );
            self.block(b).u = false( self.bsize, 1 );
        end
        
    end
    
end