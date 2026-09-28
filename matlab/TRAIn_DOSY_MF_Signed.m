function [X,D,info] = TRAIn_DOSY_MF_Signed(Y,b,D,options)
%TRAIN_DOSY_MF_SIGNED Joint TRAIn-MF on signed observations, physical SI axes.
% Y: gradients x selected signal frequencies. b: s/m^2. D: m^2/s, >=256
% increasing nodes. X=S*A is signal mass per node, NOT a density.
% Automatic rank selection uses held-out prediction and a paired one-SE rule.
% The final NNLS residual target is a separate numerical diagnostic, never
% a measured noise level or proof of chemical identification.
if nargin<4,options=struct();end
[X,D,info]=trainmf.solve(Y,b,D,options);
end
