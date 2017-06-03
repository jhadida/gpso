classdef GPSO_Search < handle
    
    properties (SetAccess = private)
        coord
        score
        found
        tol
        nns
    end
    
    properties (Transient,Dependent)
        npts
    end
    
    methods
        
        function self = GPSO_Search(varargin)
            if nargin > 0
                self.init(varargin{:});
            end
        end
        
        function n = get.npts(self)
            n = numel(self.score);
        end
        
        function init(self,coord,score,tol)
            
            if nargin < 4, tol=1e-12; end
            assert( numel(score)==size(coord,1), 'Bad input size.' );
            
            self.coord = coord;
            self.score = score;
            self.found = false(self.npts,1);
            self.tol = tol;
            self.nns = createns( coord );
            
        end
        
        function idx = find(self,x)
        % Index of closest point if its distance is within tolerance, otherwise 0.
        
            [idx,dst] = self.nns.knnsearch(x,'K',1);
            idx = idx * (dst <= self.tol);
        end
        
        function [y,idx] = getScore(self,x)
            
            idx = self.find(x);
            assert( idx > 0, 'Nearest neighbour distance above limit threshold.' );
            self.found(idx) = true;
            y = self.score(idx);
            
        end
        
    end
    
end