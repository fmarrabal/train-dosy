"""Verify all five CPU fits replay and original code/traces retain frozen hashes."""
from neural_study import *

def main():
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((OUT/'protocol.json').read_text());checks=[]
    for rel,digest in protocol['original_sources'].items():checks.append(dict(check='source unchanged',file=rel,passed=sha(ARCHIVE/rel)==digest))
    traces=[]
    for name,digest in protocol['training_trace_sha256'].items():
        p=ARCHIVE/'training'/name;checks.append(dict(check='training trace unchanged',file=name,passed=sha(p)==digest))
        with gzip.open(p,'rt',encoding='utf-8') as f:traces.append(json.load(f))
    records=[r for t in traces for r in t['records'] if r.get('validation_gain') is not None]
    feature_array=np.array([r['features'] for r in records],dtype=np.float32)
    replays=[]
    for seed in TRAIN_SEEDS:
        start=time.perf_counter();policy,history=train_policy(records,seed=seed,epochs=400,hidden=48,learning_rate=.003)
        previous=RAINetPolicy.load(OUT/'checkpoints'/f'policy_{seed}.npz')
        delta=max(float(np.max(abs(v.detach().numpy()-previous.model.state_dict()[k].detach().numpy()))) for k,v in policy.model.state_dict().items())
        expected=json.loads((OUT/'training'/f'seed_{seed}.json').read_text())
        tr=set(policy.metadata['train_problem_ids']);va=set(policy.metadata['validation_problem_ids'])
        training_mask=np.array([str(r['problem_id']) in tr for r in records])
        expected_mean=feature_array[training_mask].mean(axis=0)
        expected_scale=np.maximum(feature_array[training_mask].std(axis=0),.1)
        replay=dict(seed=seed,seconds=time.perf_counter()-start,max_abs_weight_difference=delta,
            normalization_exact=bool(np.array_equal(policy.mean,previous.mean) and np.array_equal(policy.scale,previous.scale)),
            normalization_matches_training_only=bool(np.array_equal(previous.mean,expected_mean) and np.array_equal(previous.scale,expected_scale)),
            normalization_training_record_count=int(training_mask.sum()),normalization_excluded_validation_record_count=int((~training_mask).sum()),
            all_problem_count=len(tr|va),training_problem_count=len(tr),validation_problem_count=len(va),
            history_exact=history==expected['history'],train_validation_disjoint=not bool(tr&va),
            training_ids=sorted(tr),validation_ids=sorted(va),checkpoint_sha256_verified=sha(OUT/'checkpoints'/f'policy_{seed}.npz')==expected['checkpoint_sha256'])
        replay['passed']=delta==0 and replay['normalization_exact'] and replay['normalization_matches_training_only'] and replay['history_exact'] and replay['train_validation_disjoint'] and replay['checkpoint_sha256_verified']
        replays.append(replay)
    result=dict(checks=checks,training_replays=replays,all_passed=all(c['passed'] for c in checks+replays),
        original_trace_generation_seconds=sum(t['seconds'] for t in traces),training_records=len(records),training_problems=len(traces),
        historical_training_seeds=[t['seed'] for t in traces],new_test_seeds=list(range(TEST_SEED_BASE,TEST_SEED_BASE+120)),
        test_training_seeds_disjoint=not bool(set(t['seed'] for t in traces)&set(range(TEST_SEED_BASE,TEST_SEED_BASE+120))),
        runtime_note='neural_study metrics.seconds times solve_adaptive only; excludes loading checkpoint, metrics and serialization. Original trace-generation cost is reused and reported separately.')
    dump(OUT/'reproducibility_checks.json',result);print(json.dumps(dict(all_passed=result['all_passed'],records=len(records),trace_generation_seconds=result['original_trace_generation_seconds'])))

if __name__=='__main__':main()
