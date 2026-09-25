function selection=DOSY_SelectConverged(loss,stationary,ranks)
% Paired one-standard-error rule on held-out gradients. Never silently approve
% incomplete/nonstationary candidate search. Loss columns align with ranks.
assert(size(loss,2)==numel(stationary)&&numel(ranks)==numel(stationary));
valid=logical(stationary(:)')&all(isfinite(loss),1);
selection=struct('rank',NaN,'best_rank',NaN,'complete',all(valid), ...
    'valid_candidates',valid,'ranks',ranks,'loss_per_gradient',loss, ...
    'status','no_converged_candidate','eligible',false(size(valid)));
if ~any(valid),return;end
meanloss=mean(loss,1);meanloss(~valid)=Inf;[~,best]=min(meanloss);
diffs=loss-loss(:,best);se=std(diffs,0,1)/sqrt(size(loss,1));
eligible=valid&mean(diffs,1)<=se+1e-15;
selection.rank=min(ranks(eligible));selection.best_rank=ranks(best);
selection.paired_se=se;selection.eligible=eligible;
if all(valid),selection.status='selected_converged';else,selection.status='unresolved_candidate_optimization';end
end
