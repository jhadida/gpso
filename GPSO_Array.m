classdef GPSO_Array < handle
    
    properties
        x, n, b;
    end
    properties (Transient,Dependent)
        nrows,ncols;
    end
    
    methods
        
        function self = GPSO_Array(varargin)
            self.clear();
            if nargin > 0
                self.create( n, c, v );
            end
        end
        
        function n = get.nrows(self), n=size(self.x,1); end
        function n = get.ncols(self), n=size(self.x,2); end
        
        function self=clear(self)
            self.x = [];
            self.n = 0;
            self.b = 1;
        end
        
        function self=create(self,r,c)
            self.x = nan(r,c);
            self.n = 0;
            self.b = r;
        end
        
        function disp(self)
            disp(self.x(1:self.n,:));
        end
        
        function k=append(self,v)
            k = self.n+1;
            if k > self.nrows
                self.x = vertcat( self.x, nan(self.b,self.ncols) );
            end
            self.n = k;
            self.x(k,:) = v;
        end
        
        function self=setr(self,r,v)
            assert( r <= self.n, 'Index out of bounds.' );
            self.x(r,:) = v;
        end
        function x=getr(self,r)
            assert( all(r(:) <= self.n), 'Index out of bounds.' );
            x = self.x(r,:);
        end
        
    end
    
end