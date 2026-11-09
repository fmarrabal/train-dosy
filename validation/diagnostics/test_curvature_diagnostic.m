function results=test_curvature_diagnostic(outputFile)
% Three deterministic algebraic tests; no data fitting or solver modification.
here=fileparts(mfilename('fullpath'));addpath(here);
D=logspace(log10(.1e-9),log10(15e-9),256)';
b=linspace(0,2.2e9,24)';A=[1 0 .7;.2 1 .3];
[independent,L,v]=curvature_diagnostic(b,D,A,1e-8);
assert(independent.conditional_injectivity_numerically_supported);
assert(independent.amplitude_numeric_row_rank==2&&independent.operator_consistent);
duplicate=[1 .7 .4;1 .7 .4];
[dependent,~,~]=curvature_diagnostic(b,D,duplicate,1e-8);
assert(~dependent.conditional_injectivity_numerically_supported);
assert(dependent.amplitude_numeric_row_rank==1);
U=[v,-v];K=exp(-b*D');
assert(norm(K*U*duplicate,'fro')<1e-10&&norm(L*U,'fro')<1e-7);
constant=curvature_diagnostic(zeros(size(b)),D,A,1e-8);
assert(~constant.conditional_injectivity_numerically_supported);
assert(constant.kernel_null_direction_ratio<1e-10);
results=struct('passed',true,'cases',3,'independent_amplitudes',independent, ...
    'duplicate_amplitudes',dependent,'constant_kernel',constant, ...
    'matlab_version',version,'scope','Synthetic algebraic checks; not runtime instrumentation or experimental validation');
if nargin>0
    fid=fopen(outputFile,'w');assert(fid>=0);
    closeFile=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s',jsonencode(results,PrettyPrint=true));
end
fprintf('PASS: independent rows, duplicated rows, and constant-kernel degeneracy.\n');
end
