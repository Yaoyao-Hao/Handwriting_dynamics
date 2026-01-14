%% Orth Subspace fig.2d
clc
clear all

currpath = pwd;
% addpath([currpath '/optFunctions']);

example_data = 'PCA_0623.mat';
data = load(example_data);
C_stroke = data.C_stroke; % covariance matrix
C_penlift = data.C_penlift;

d_st=4; % dim of Orth-subspace
d_pl=4;

[Q, ~, info, options] = orthogonal_subspaces(C_stroke,d_st,C_penlift,d_pl); 
P1 = [eye(d_st); zeros(d_pl,d_st)];
P2 = [zeros(d_st, d_pl); eye(d_pl)];
dmax = max(d_st,d_pl);
eigvals1 = eigs(C_stroke, dmax, 'la'); 
eigvals2 = eigs(C_penlift, dmax, 'la');

% variance explained for each condition in each subspace
st_on_st = var_proj(Q*P1,C_stroke,sum(eigvals1(1:d_st)));
pl_on_st = var_proj(Q*P1,C_penlift,sum(eigvals2(1:d_st))); 
pl_on_pl = var_proj(Q*P2,C_penlift,sum(eigvals2(1:d_pl)));
st_on_pl = var_proj(Q*P2,C_stroke,sum(eigvals1(1:d_pl)));
