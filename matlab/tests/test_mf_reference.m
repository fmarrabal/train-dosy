function result = test_mf_reference()
% Compare the manuscript MF-AUTO implementation with its saved one-factor fit.
here=fileparts(mfilename('fullpath'));matlabDir=fileparts(here);root=fileparts(matlabDir);
addpath(matlabDir);
expected=load(fullfile(root,'examples','one_component','MF_AUTO.mat'));
actual=example_mf();
input=jsondecode(fileread(fullfile(root,'examples','one_component','input.json')));
assert(isequal(size(actual.X),size(expected.X)) && size(actual.X,1)==256);
assert(isequal(actual.D(:),expected.D(:)) && isequal(actual.ppm(:),expected.ppm(:)));
assert(actual.info.selected_rank==expected.info.selected_rank && actual.info.resolved_only);
K=exp(-input.b(:)*actual.D(:)');
result=struct('selected_rank',actual.info.selected_rank,'bins',numel(actual.D), ...
    'selected_frequencies',size(actual.X,2),'max_absolute_X_difference',max(abs(actual.X-expected.X),[],'all'), ...
    'relative_prediction_difference',norm(K*(actual.X-expected.X),'fro')/norm(K*expected.X,'fro'));
assert(all(abs(actual.X-expected.X)<=1e-12+1e-10*abs(expected.X),'all'), ...
    'TRAInMFTest:Reference','MF-AUTO bin masses differ from the saved fit.');
assert(result.relative_prediction_difference<=1e-10);
fprintf('MF-AUTO parity: %s\n',jsonencode(result));
end
