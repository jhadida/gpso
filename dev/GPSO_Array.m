classdef GPSO_Array < handle
%
% Array with reallocation by block to allow appending with reduced overhead.
%

    properties
        x; % data matrix
        n; % number of rows currently used
        b; % block size
    end
    properties (Transient,Dependent)
        nrows, ncols, maxrows;
    end
    
    methods
        
        function self = GPSO_Array(varargin)
            self.clear();
            if nargin > 0
                self.create( n, c, v );
            end
        end
        
        function n = get.nrows(self), n=self.n; end
        function n = get.ncols(self), n=size(self.x,2); end
        function n = get.maxrows(self), n=size(self.x,1); end
        
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
        
        function x=data(self)
            x = self.x( 1:self.n, : );
        end
        function disp(self)
            disp(self.data);
        end
        
        function k=append(self,v)
            
            m = size( v, 1 ); % #of rows to append
            f = self.n + 1; % index of first appended row
            l = self.n + m; % index of last appended row
                        
            % allocate more rows if needed
            while l > self.maxrows
                self.x = vertcat( self.x, nan(self.b,self.ncols) );
            end
            
            % append
            k = f:l;
            self.n = l;
            self.x(k,:) = v;
        end
        
        function self=shrink_to_fit(self)
            self.x = self.data();
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