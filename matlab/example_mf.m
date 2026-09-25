function result=example_mf()
% Reproducible one-factor example. Run from any directory.
here=fileparts(mfilename('fullpath'));root=fileparts(here);
z=jsondecode(fileread(fullfile(root,'examples','one_component','input.json')));
assert(isfield(z,'mask'),'Example mask was generated without held-out rows.');
Y=z.Y(:,z.mask);b=z.b(:);ppm=z.ppm(z.mask);sigma=z.sigma;
D=logspace(log10(z.diffusion_bounds(1)),log10(z.diffusion_bounds(2)),z.bins)';
addpath(fullfile(here,'reference'));
[~,order]=sort(b);val=false(numel(b),1);val(order(3:4:end-1))=true;
[X,D,info]=DOSY_MF_Auto(Y,b,D,sigma,~val,val,struct());
assert(~isempty(X)&&all(isfinite(X),'all')&&min(X,[],'all')>=-1e-12);
result=struct('X',X,'D',D,'ppm',ppm,'info',info);
fprintf('MF status=%s, predictive factors=%d, bins=%d\n',info.status,info.selected_rank,numel(D));
end
