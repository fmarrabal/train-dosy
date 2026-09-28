function [X,D_space,info] = TRAIn_DOSY_MF(Z,b,D_params,options)
%TRAIN_DOSY_MF Revised entry point. b MUST be physical b values in s/m^2.
% D_params=[Dmin Dmax bins], D in 1e-9 m^2/s, bins>=256.
% Use signal_mask to select complete signal regions from the full spectrum.
% Historical V3/V3.1 preprocessing is deliberately not reproduced here.
% For a direct SI grid use TRAIn_DOSY_MF_Signed instead.
if nargin<4,options=struct();end
validateattributes(D_params,{'numeric'},{'real','finite','vector','numel',3});
assert(D_params(1)>0&&D_params(2)>D_params(1)&&D_params(3)>=256&&D_params(3)==fix(D_params(3)), ...
    'TRAInMF:Grid','D_params must be [positive min, larger max, integer bins >=256].');
D_space=logspace(log10(D_params(1)),log10(D_params(2)),D_params(3))';
assert(isstruct(options)&&isscalar(options),'TRAInMF:Options','Scalar options struct required.');
% Accept safe old settings explicitly; reject options that would silently
% reinstate a different objective or intensity preprocessing.
safe=struct('signal_mode','real','normalize_y',false,'auto_scale_G2',false, ...
    'G2_scale',1,'A_solver','lsqnonneg','lambda_A',0,'lambda_S_smooth',0, ...
    'lambda_S_sparse',0,'auto_lambda',false,'spectral_smooth_window',1);
names=fieldnames(safe);
for j=1:numel(names)
    name=names{j};
    if isfield(options,name)
        assert(isequal(options.(name),safe.(name))|| ...
            (isnumeric(safe.(name))&&isequal(double(options.(name)),double(safe.(name)))), ...
            'TRAInMF:LegacyOption','%s is incompatible with signed TRAIn-MF. Use the documented revised options.',name);
        options=rmfield(options,name);
    end
end
if isfield(options,'auto_r')
    if options.auto_r,options.n_components='auto';end
    options=rmfield(options,'auto_r');
end
mask=true(1,size(Z,2));
if isfield(options,'signal_mask')
    mask=options.signal_mask;
    assert(islogical(mask)&&isvector(mask)&&numel(mask)==size(Z,2)&&any(mask), ...
        'TRAInMF:Mask','signal_mask must be logical, match columns and retain signal.');
    mask=mask(:)';options=rmfield(options,'signal_mask');
end
[selected,~,info]=TRAIn_DOSY_MF_Signed(Z(:,mask),b,D_space*1e-9,options);
% NaN means not estimated, not a measured absence of intensity.
X=nan(numel(D_space),size(Z,2));X(:,mask)=selected;
info.process_idx=find(mask);info.freq_mask=mask;info.n_processed=nnz(mask);
info.excluded_frequencies='NaN: not estimated';
end
