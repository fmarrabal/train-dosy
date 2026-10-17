function run_mf_batch(ranks)
root=fileparts(mfilename('fullpath'));cfg=jsondecode(fileread(fullfile(root,'config.json')));
for rank=ranks(:)'
 for snr=cfg.snrs(:)'
  for warm=cfg.warm_sweeps(:)'
   id=sprintf('rank%d_snr%d',rank,snr);
   run_mf_case(id,warm);
  end
 end
end
end
