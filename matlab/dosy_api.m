function result=dosy_api(request,baseUrl)
% API client: signed Y(nGradient,nFrequency), SI b, ppm, positive sigma.
% TRAIn-MF uses the signed MATLAB solver; MF-AUTO uses the manuscript reference.
% RAI-S and DOME-S run in the Python backend.
if nargin<2,baseUrl='http://127.0.0.1:8765';end
opts=weboptions('MediaType','application/json','Timeout',660);
result=webwrite([baseUrl '/v1/fit'],request,opts);
end
