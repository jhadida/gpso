function gpml_start()
%
% gpml_start()
%
% Add GPML sources to the path.
% This is used internally by gpso_run and should not be called manually.
%
% JH

    if isempty(which('gp'))
        dirs = gpml_path();
        addpath(dirs{:});
    end

end
