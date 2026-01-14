%% Exclusive Subspace and Shared Subspace fig.2f
clc
clear all

currpath = pwd;
addpath([currpath '/optFunctions']);

example_data = 'PCA_0623.mat';
data = load(example_data);
C_stroke = data.C_stroke; % covariance matrix
C_penlift = data.C_penlift;

d_st=4;d_pl=4;% dim of Excl-subspace
alphaNullSpace = 0.01;
[Q_st,flagSt] = exclusive_subspace(C_stroke,C_penlift,d_st,alphaNullSpace); 
eigvals1 = eigs(C_stroke, d_st, 'la');
stroke_var_on_excl = var_proj(Q_st,C_stroke,sum(eigvals1(1:d_st)));

[Q_pl,flagPl] = exclusive_subspace(C_penlift,C_stroke,d_pl,alphaNullSpace);
eigvals2 = eigs(C_penlift, d_pl, 'la');
penlift_var_on_excl = var_proj(Q_pl,C_penlift,sum(eigvals2(1:d_pl)));

stroke_var_on_excl_penlift = var_proj(Q_pl,C_stroke,sum(eigvals1(1:d_st)));
penlift_var_on_excl_stroke = var_proj(Q_st,C_penlift,sum(eigvals2(1:d_pl)));


d_shared=4;% dim of Shared-subspace
[Q1,Qshared, Qcost, info, options] = shared_subspace(Q_st,d_st,Q_pl,d_pl,C_stroke,C_penlift,d_shared);
dmax = max(max(d_st,d_pl),d_shared);
eigvals1 = eigs(C_stroke, dmax, 'la');
eigvals2 = eigs(C_penlift, dmax, 'la');
% variance explained for each condition in the shared subspace
stroke_on_shared = var_proj(Qshared,C_stroke,sum(eigvals1(1:d_shared)));
penlift_on_shared = var_proj(Qshared,C_penlift,sum(eigvals2(1:d_shared)));

