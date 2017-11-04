function [dirs,root] = gpml_path()
%
% dirs = gpml_path()
%
% Returns cell-array of folders containing GPML sources.
% Used internally by gpso_run.
%
% JH

    root = fileparts(mfilename('fullpath'));
    gpml = fullfile(root,'gpml');
    dirs = dk.mapfun( @(x) fullfile(gpml,x), {'cov','doc','inf','lik','mean','prior','util/lbfgsb','util'}, false );
    dirs = [ {gpml}, dirs ];

end
