function [x,residual] = nnls(K,y)
%NNLS Use installed MATLAB lsqnonneg; fail instead of substituting a solver.
persistent verifiedPath
currentPath=which('lsqnonneg');
if ~strcmp(currentPath,verifiedPath)
    assert(startsWith(currentPath,[matlabroot filesep],'IgnoreCase',true), ...
        'TRAInMF:NNLS','Installed MATLAB lsqnonneg is required; remove shadowing files.');
    verifiedPath=currentPath;
end
[x,~,~,flag]=lsqnonneg(K,y,optimset('Display','off','MaxIter',3*size(K,2)));
assert(flag==1,'TRAInMF:NNLS','MATLAB lsqnonneg did not converge.');
residual=norm(K*x-y);
end
