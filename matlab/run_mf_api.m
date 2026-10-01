function run_mf_api(inputFile,outputFile)
% Explicit revised TRAIn-MF or frozen MF-AUTO; both use physical SI axes.
here=fileparts(mfilename('fullpath'));addpath(fullfile(here,'reference'));
z=load(inputFile);b=z.b(:);D=z.D(:);Y=z.Y;
[~,order]=sort(b);val=false(numel(b),1);val(order(3:4:end-1))=true;
if isfield(z,'method')&&strcmp(z.method,'TRAIn-MF')
    % Match Python discovery's excluded rows; no validation data in masking.
    options=struct('r_max',double(z.max_components),'validation_rows',val,'sigma',double(z.sigma));
    [X,D,info]=TRAIn_DOSY_MF_Signed(Y,b,D,options);
    out=struct('selected_rank',info.selected_rank,'active_rank',info.active_rank,'status',info.status, ...
        'success',info.success,'X',X,'D_grid',D,'A',info.A,'S',info.S, ...
        'kkt',[],'chemical_species_identified',false,'protocol',info.protocol, ...
        'diagnostics',rmfield(info,{'S','A'}));
else
[X,D,info]=DOSY_MF_Auto(Y,b,D,z.sigma,~val,val,struct());
out=struct('selected_rank',info.selected_rank,'status',info.status, ...
    'success',info.resolved_only,'X',X,'D_grid',D,'A',[],'S',[], ...
    'kkt',[],'chemical_species_identified',false);
if isfield(info,'fit')
    out.A=info.fit.A;out.S=info.fit.S;out.kkt=info.fit.kkt.relative;
end
end
f=fopen(outputFile,'w');assert(f>=0);c=onCleanup(@()fclose(f));
fprintf(f,'%s',jsonencode(out));
end
