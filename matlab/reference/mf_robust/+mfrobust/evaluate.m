function [val,parts,k]=evaluate(K,Y,S,A,L,o)
% Loss and A ridge averaged over frequency. KKT scales do not grow with nf.
nf=size(Y,2);B=K*S;E=B*A-Y;
parts=[sum(E.^2,'all')/(2*nf),o.lambda_S*sum((L*S).^2,'all')/2,o.lambda_A*sum(A.^2,'all')/(2*nf)];val=sum(parts);
if nargout<3,return;end
gS=K'*(E*A'/nf)+o.lambda_S*(L'*(L*S));nu=sum(S.*gS,1);dualS=gS-nu;
scaleS=max(1,max(abs(K'*(Y*A'/nf)),[],1));
rawS=max(max(max(-dualS,0),[],1),max(abs(S.*dualS),[],1));
G=B'*Y;gA=B'*E+o.lambda_A*A;scaleA=max(1,max(abs(G),[],1));
rawA=max(max(max(-gA,0),[],1),max(abs(A.*gA),[],1)./max(1,max(A,[],1)));
primal=max([max(abs(sum(S,1)-1)),max(-S,[],'all'),max(-A,[],'all'),0]);
k=struct('S',max(rawS./scaleS),'A',max(rawA./scaleA),'primal',primal, ...
    'relative',max([max(rawS./scaleS),max(rawA./scaleA),primal]));
end
