function result=dosy_api(request,baseUrl)
% API client: signed Y(nGradient,nFrequency), SI b, ppm, positive sigma.
% Native MF lives in reference/DOSY_MF_Auto; RAI-S/DOME-S use Python backend.
if nargin<2,baseUrl='http://127.0.0.1:8765';end
opts=weboptions('MediaType','application/json','Timeout',660);
result=webwrite([baseUrl '/v1/fit'],request,opts);
end
