function run_mf_api(inputFile,outputFile)
% Native frozen MF. SI b and D are conjugate; no implicit b conversion.
here=fileparts(mfilename('fullpath'));addpath(fullfile(here,'reference'));
z=load(inputFile);b=z.b(:);D=z.D(:);Y=z.Y;
[~,order]=sort(b);val=false(numel(b),1);val(order(3:4:end-1))=true;
[X,D,info]=DOSY_MF_Auto(Y,b,D,z.sigma,~val,val,struct());
out=struct('selected_rank',info.selected_rank,'status',info.status, ...
    'success',info.resolved_only,'X',X,'D_grid',D,'A',[],'S',[], ...
    'kkt',[],'chemical_species_identified',false);
if isfield(info,'fit')
    out.A=info.fit.A;out.S=info.fit.S;out.kkt=info.fit.kkt.relative;
end
f=fopen(outputFile,'w');assert(f>=0);c=onCleanup(@()fclose(f));
fprintf(f,'%s',jsonencode(out));
end
