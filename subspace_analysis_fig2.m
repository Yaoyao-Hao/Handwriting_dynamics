%% PCA fig.2ab
clc
clear all

currpath = pwd;
% addpath([currpath '/optFunctions']);

example_data = 'subspace_data0623.mat';
data = load(example_data);
stroke = data.stroke_data'; % NxTC
penlift = data.penlift_data';

dim = 4; % num of PCs
[coeff_s, score_s, latent_s,~, explained_s] = pca(stroke, 'Centered', true); 
stroke_pc = coeff_s(:, 1:dim);
% sum_explained_s = sum(explained_s(1:dim))
[coeff_p, score_p, latent_p,~, explained_p] = pca(penlift, 'Centered', true);
penlift_pc = coeff_p(:, 1:dim);
% sum_explained_c = sum(explained_c(1:dim))

% stroke projection on penlift PCs
stroke_proj = stroke * penlift_pc ;
stroke_var = sum(var(stroke)); 
stroke_proj_var = var(stroke_proj);
stroke_proj_var_ratio = stroke_proj_var/stroke_var;
% penlift projection on stroke PCs
penlift_proj = penlift * stroke_pc ;
penlift_var = sum(var(penlift));
penlift_proj_var = var(penlift_proj);
penlift_proj_var_ratio = penlift_proj_var/penlift_var;

% normalized variance captured
C_stroke = cov(stroke); % covariance matrix of stroke
[eigvec_s, eigval_matrix] = eig(C_stroke); 
[sorted_eigval_s, idx_s] = sort(diag(eigval_matrix), 'descend'); 
sorted_eigval_s(find(sorted_eigval_s<0))=0;
A_stroke_on_penlift = trace(penlift_pc'*C_stroke*penlift_pc)/sum(sorted_eigval_s(1:dim));

C_penlift = cov(penlift); % covariance matrix of stroke
[eigvec_p, eigval_matrix] = eig(C_penlift); 
[sorted_eigval_p, idx_p] = sort(diag(eigval_matrix), 'descend');
sorted_eigval_p(find(sorted_eigval_p<0))=0;
A_penlift_on_stroke = trace(stroke_pc'*C_penlift*stroke_pc)/sum(sorted_eigval_p(1:dim));

% random
sorted_eigvec_s = eigvec_s(:,idx_s);  % re-sort
sorted_eigvec_p = eigvec_c(:,idx_p); 
stroke_lamda = diag(sorted_eigval_s);
penlift_lamda = diag(sorted_eigvec_p);

n_samples=10000;
A_rnd_samples = [];
for i=1:n_samples
    v = randn(size(stroke,2), dim);
    Q_stroke_rnd = orth((sorted_eigvec_s*sqrt(stroke_lamda)*v)/norm(sorted_eigvec_s*sqrt(stroke_lamda)*v));
    Q_penlift_rnd = orth((sorted_eigvec_p*sqrt(penlift_lamda)*v)/norm(sorted_eigvec_p*sqrt(penlift_lamda)*v));

    A_rnd = trace(Q_stroke_rnd'*Q_penlift_rnd*Q_penlift_rnd'*Q_stroke_rnd)/dim;
    A_rnd_samples = [A_rnd_samples A_rnd];
end

p1 = mean(A_rnd_samples>=A_stroke_on_penlift);
p2 = mean(A_rnd_samples>=A_penlift_on_stroke);
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

