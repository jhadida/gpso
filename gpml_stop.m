function gpml_stop()
%
% gpml_stop()
%
% Remove GPML sources from the path.
% This is used internally by gpso_run and should not be called manually.
%
% JH

    if isempty(which('gp'))
        return;
    end

    dirs = gpml_path();
    rmpath(dirs{:});

end
