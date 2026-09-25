function p = prepare(Y,b,D,options)
% Validate the physical/statistical contract. No estimated baseline or units.
if nargin<4, options=struct(); end
defaults=struct('units','','sigma',[],'baseline',[],'signal_scale',[], 'n_components',1, ...
    'lambda_S',1e-4,'lambda_A',1e-8,'lambda_frequency',0, ...
    'regularizer_order',2,'frequency_axis',[],'spectral_regions',[], ...
    'max_spectral_gap',[],'n_starts',4,'seed',23092026, ...
    'max_outer_iter',300,'objective_tol',1e-9,'kkt_tol',1e-6, ...
    'solver_tol',1e-10,'solver_max_iter',2000, ...
    'initial_S',[],'initial_A',[],'verbose',false, ...
    'component_similarity_threshold',0.9995,'active_component_relative_tol',1e-8);
assert(isstruct(options)&&isscalar(options),'DOSY:Options','Options must be a scalar struct.');
names=fieldnames(options);
for i=1:numel(names)
    assert(isfield(defaults,names{i}),'DOSY:UnknownOption','Unknown option: %s',names{i});
    defaults.(names{i})=options.(names{i});
end
o=defaults;
assert(any(strcmp(o.units,{'SI','conjugate'})),'DOSY:Units', ...
    'Specify units=''SI'' (b in s/m^2, D in m^2/s) or ''conjugate'' explicitly.');
assert(isnumeric(Y)&&ismatrix(Y)&&~isempty(Y)&&isreal(Y)&&all(isfinite(Y),'all'), ...
    'DOSY:Data','Y must be a finite, real nGradient-by-nFrequency matrix; phase complex data externally.');
assert(isnumeric(b)&&isvector(b)&&isnumeric(D)&&isvector(D),'DOSY:Coordinates','b and D must be numeric vectors.');
Y=double(Y); b=double(b(:)); D=double(D(:)); [ng,nf]=size(Y); nd=numel(D);
assert(isreal(b)&&numel(b)==ng&&all(isfinite(b))&&all(b>=0), ...
    'DOSY:Gradients','b must be finite, nonnegative and match the rows of Y.');
assert(isreal(D)&&nd>=2&&all(isfinite(D))&&all(D>0)&&all(diff(D)>0), ...
    'DOSY:Grid','Supply at least two strictly increasing positive D nodes, not [min max n].');
for name={'n_components','n_starts','max_outer_iter','solver_max_iter'}
    v=o.(name{1}); assert(isnumeric(v)&&isreal(v)&&isscalar(v)&&isfinite(v)&&v>=1&&v==round(v), ...
        'DOSY:Options','%s must be a positive integer.',name{1});
end
assert(o.n_components<=min(nd,nf),'DOSY:Rank','r must not exceed min(nD,nFrequency).');
for name={'lambda_S','lambda_A','lambda_frequency'}
    v=o.(name{1}); assert(isnumeric(v)&&isreal(v)&&isscalar(v)&&isfinite(v)&&v>=0, ...
        'DOSY:Options','%s must be finite and nonnegative.',name{1});
end
for name={'objective_tol','kkt_tol','solver_tol','active_component_relative_tol'}
    v=o.(name{1}); assert(isnumeric(v)&&isreal(v)&&isscalar(v)&&isfinite(v)&&v>0, ...
        'DOSY:Options','%s must be finite and positive.',name{1});
end
assert(isscalar(o.regularizer_order)&&any(o.regularizer_order==[1,2]),'DOSY:Options','regularizer_order must be 1 or 2.');
assert(isscalar(o.seed)&&isfinite(o.seed)&&o.seed>=0&&o.seed==floor(o.seed)&&o.seed<2^32,'DOSY:Seed','Invalid seed.');
assert(isscalar(o.verbose)&&(islogical(o.verbose)||ismember(o.verbose,[0 1])),'DOSY:Options','verbose must be logical.');
assert(isscalar(o.component_similarity_threshold)&&isfinite(o.component_similarity_threshold)&& ...
    o.component_similarity_threshold>0&&o.component_similarity_threshold<=1,'DOSY:Options','Invalid similarity threshold.');
baseline=expand_array(o.baseline,ng,nf,0,'baseline');
signal=Y-baseline; scale=norm(signal,'fro')/sqrt(numel(signal));
assert(isfinite(scale),'DOSY:Scale','Signal scale overflow. Rescale in documented physical units.');
if scale==0, scale=1; end
if ~isempty(o.signal_scale)
    assert(isnumeric(o.signal_scale)&&isreal(o.signal_scale)&&isscalar(o.signal_scale)&& ...
        isfinite(o.signal_scale)&&o.signal_scale>0,'DOSY:Scale','signal_scale must be finite and positive.');
    scale=o.signal_scale;
end
if isempty(o.sigma)
    sigma=scale*ones(ng,nf); noise_known=false;
else
    sigma=expand_array(o.sigma,ng,nf,1,'sigma'); noise_known=true;
    assert(all(sigma>0,'all'),'DOSY:Noise','sigma must be strictly positive.');
end
weights=scale./sigma;
assert(all(isfinite(weights.^2),'all'),'DOSY:Noise','Weights overflow; check data and noise scales.');
[Lu,q,Lf,edges]=dosyv4.regularizers(D,nf,o);
K=exp(-b*D');
o.sigma=sigma; o.baseline=baseline;
if ~isempty(o.initial_S)
    assert(isequal(size(o.initial_S),[nd,o.n_components])&&isreal(o.initial_S)&& ...
        all(isfinite(o.initial_S),'all')&&all(o.initial_S>=0,'all')&& ...
        max(abs(sum(o.initial_S,1)-1))<1e-10,'DOSY:InitialS','initial_S must have feasible unit-mass columns.');
end
if ~isempty(o.initial_A)
    assert(isequal(size(o.initial_A),[o.n_components,nf])&&isreal(o.initial_A)&& ...
        all(isfinite(o.initial_A),'all')&&all(o.initial_A>=0,'all'), ...
        'DOSY:InitialA','initial_A must be nonnegative, finite and in raw signal units.');
end
p=struct('Y',signal/scale,'raw_Y',Y,'b',b,'D',D,'K',K,'W',weights, ...
    'sigma',sigma,'noise_known',noise_known,'baseline',baseline,'scale',scale, ...
    'L',Lu,'quadrature',q,'Lf',Lf,'edges',edges,'options',o);
end

function x=expand_array(x,ng,nf,default,name)
if isempty(x), x=default*ones(ng,nf); end
assert(isnumeric(x)&&isreal(x)&&all(isfinite(x),'all'),'DOSY:Array','%s must be finite and real.',name);
if isscalar(x), x=repmat(double(x),ng,nf);
elseif isequal(size(x),[1,nf]), x=repmat(double(x),ng,1);
elseif ~isequal(size(x),[ng,nf]), error('DOSY:Array','%s must be scalar, 1-by-nFrequency, or same size as Y.',name);
else, x=double(x); end
end
