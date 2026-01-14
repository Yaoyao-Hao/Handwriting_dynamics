%% shared neural dynamics  fig.3a
clc
clear all
currpath = pwd;
% addpath([currpath '/utils/TBFandCondSpecificDynamics']);
% addpath([currpath '/utils/evalFit']);
% addpath([currpath '/TBF']);

example_data = '0623_mua_aver_smo.mat';
data = load(example_data);
stroke = data.Stroke_cell;
penlift = data.Penlift_cell;

stroke_data = struct(); % cell to struct
for i = 1:size(stroke,1)
    stroke_data(i).matrix = stroke{i};
end
penlift_data = struct();
for i = 1:size(penlift,1)
    penlift_data(i).matrix = penlift{i};
end

dimUse=5; % num of neural dynamics
[basisFxns,loadings,params] = eigTransform([stroke_data,penlift_data],dimUse);

% variance explained for each stroke and pen lift
example_data = '0623_mua_noaver_smo.mat';
data = load(example_data);
stroke = data.Stroke_cell;
penlift = data.Penlift_cell;

for cond = 1:size(stroke,1)
    stroke_data(cond).matrix = stroke{cond};
end
for cond = 1:size(penlift,1)
    penlift_data(cond).matrix = penlift{cond};%
end
% reconstruction
act_stroke = struct();
for cond = 1:size(stroke,1)
    act_stroke(cond).matrix = stroke_data(cond).matrix*pinv(basisFxns.');
end
stroke_approx = struct();
for cond = 1:size(act_stroke,2)
    stroke_approx(cond).matrix = act_stroke(cond).matrix*basisFxns';
end

act_penlift = struct();
for cond = 1:size(penlift,1)
    act_penlift(cond).matrix = penlift_data(cond).matrix*pinv(basisFxns.');
end
penlift_approx = struct();
for cond = 1:size(act_penlift,2)
    penlift_approx(cond).matrix = act_penlift(cond).matrix*basisFxns';
end
% variance explained
VE_stroke = getVarExplained(stroke_data,stroke_approx,'ind');
VE_penlift = getVarExplained(penlift_data,penlift_approx,'ind');
act = [act_stroke act_penlift];
