function report=audit_saved_mf_reference(outputFile)
% Read-only extraction from the unchanged manuscript illustration.
% This is not a new inversion or a sweep-count/SNR convergence experiment.
here=fileparts(mfilename('fullpath'));addpath(here);
paperRoot=fileparts(fileparts(here));
source=fullfile(paperRoot,'reproduction_inputs','ilt_unified_benchmark_20260924', ...
    'results','r3_snr200_rep1_mf_auto.mat');
s=load(source);f=s.info;records=cell(1,numel(f.starts));
dx=diff(log(f.D(:)));
weights=[dx(1)/2;(dx(1:end-1)+dx(2:end))/2;dx(end)/2];
quadratureDifference=max(abs(weights-f.quadrature(:)));
assert(quadratureDifference<1e-12,'CurvatureAudit:Weights','Saved weights differ from the stated operator.');
for j=1:numel(f.starts)
    a=f.starts(j);
    before=a.kkt_history(end);
    record=struct('start',j,'alternating_sweeps',a.iterations, ...
        'post_alternating_kkt',before,'final_kkt',a.kkt.relative, ...
        'final_status',a.status,'final_objective',a.objective);
    if numel(a.history)>=1+2*a.iterations
        record.post_alternating_objective=a.history(1+2*a.iterations);
    end
    if isfield(a,'profile_refinement')&&~isempty(fieldnames(a.profile_refinement))
        p=a.profile_refinement;
        record.sqp_iterations=p.iterations;record.sqp_evaluations=p.evaluations;
        record.sqp_exitflag=p.exitflag;record.sqp_accepted=p.accepted;
    end
    records{j}=record;
end
postHoc=curvature_diagnostic(f.x,f.D,f.A,f.options.lambda_S);
report=struct('source',source,'source_solver_version',f.version, ...
    'source_status',f.status,'source_selected_rank',size(f.A,1), ...
    'starts',{records},'post_hoc_curvature',postHoc, ...
    'saved_quadrature_max_abs_difference',quadratureDifference, ...
    'source_elapsed_seconds',f.elapsed_seconds, ...
    'scope','Read-only extraction plus post-hoc curvature check of one saved illustration; not a convergence study');
if nargin>0
    fid=fopen(outputFile,'w');assert(fid>=0);
    closeFile=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));
end
disp(report);
end
