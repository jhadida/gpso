function [best,T] = explore(self,node,depth,varsigma,until)
%
% node: either a 1x2 array [h,i] (cf get_node), or a node structure
% depth: how deep the exploration tree should be 
%   (WARNING: tree grows exponentially, there will be 3^depth node)
% varsigma: optimism constant to be used locally for UCB
% until: early cancelling criterion (stop if one of the samples has a better score)
%   (NOTE: default is Inf, so full exploration)
%
% Explore node by growing exhaustive partition tree, using surrogate for evaluation.
%

    if nargin < 5, until=inf; end

    % get node if index were passed
    if ~isstruct(node)
        node = self.get_node(node(1),node(2));
    end

    % Temporary exploration tree
    T = dk.struct.repeat( {'lower','upper','coord','samp'}, depth+1, 1 );

    T(1).lower = node.lower;
    T(1).upper = node.upper;
    T(1).coord = node.coord;
    T(1).samp  = node.samp;

    best = T(1).samp;
    if best(3) >= until, return; end

    for h = 1:depth
        for i = 1:3^(h-1)

            % evaluate GP
            [g,d,x,s]  = split_largest_dimension( T(h), i, T(h).coord(i,:) );
            [mu,sigma] = self.srgt.gp_call( [g;d] );

            % update best score
            ucb   = mu + varsigma*sigma;
            [u,k] = max(ucb);
            if u > best(3)
                best = [ mu(k), sigma(k), u ];
            end

            if u >= until; break; end % early cancelling

            % record new nodes
            U = split_tree( T(h), i, g, d, x, s );
            T(h+1).coord = [ T(h+1).coord; U.coord ];
            T(h+1).lower = [ T(h+1).lower; U.lower ];
            T(h+1).upper = [ T(h+1).upper; U.upper ];
            T(h+1).samp  = [ T(h+1).samp; [mu,sigma,ucb] ];

        end
        if u >= until; break; end % chain-break
    end

end