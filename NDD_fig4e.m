%% neural dynamic decoding fig.4e
clc
clear all
currpath = pwd;
% addpath([currpath '/FittingDynamicsToSingleConditions']);
% addpath([currpath '/utils/TBFandCondSpecificDynamics']);
% addpath([currpath '/utils/evalFit']);
% addpath([currpath '/utils/subspaceUtils']);
% addpath([currpath '/TBF']);

example_data = 'mua_0623.mat';
data = load(example_data);
trial_velocity = data.trial_velocity;
break_ind = data.break_ind;
trial_target = data.trial_target;
trial_mask = data.trial_mask;
trial_breakNum = data.trial_breakNum; 

example_data = '0623_mua_aver_smo_multi.mat'; % trial-averaged
data = load(example_data);
stroke_aver = data.Stroke_cell;
penlift_aver = data.Penlift_cell;
stroke_kin_aver = data.Stroke_kin_cell; 
penlift_kin_aver = data.Penlift_kin_cell;
stroke_ind_aver =  data.stroke_ind;
penlift_ind_aver = data.penlift_ind;

example_data = '0623_mua_noaver_smo_multi.mat';% single trial
stroke = data.Stroke_cell;
penlift = data.Penlift_cell;
stroke_kin = data.Stroke_kin_cell;
penlift_kin = data.Penlift_kin_cell;
stroke_ind =  data.stroke_ind;
penlift_ind = data.penlift_ind;

stroke_kin_tmp = stroke_kin(:,1); 
penlift_kin_tmp = penlift_kin(:,1);
stroke_kin_aver_tmp = stroke_kin_aver(:,1); 
penlift_kin_aver_tmp = penlift_kin_aver(:,1);

% Vxy
kin_dim = [1,2];
Vxy = NDD(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            dim, kin_dim);
% Vz
kin_dim = 3;
Vz = NDD(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            dim, kin_dim);
% Grip
kin_dim = 4;
Grip = NDD(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            dim, kin_dim);
% pressure
kin_dim = 5;
Pressure = NDD(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            dim, kin_dim);
% EMG 6 subjects
EMG = struct();
kin_dim = 6:13;
for sub=1:6
    stroke_kin_tmp = stroke_kin(:,sub);
    penlift_kin_tmp = penlift_kin(:,sub);
    stroke_kin_aver_tmp = stroke_kin_aver(:,sub);
    penlift_kin_aver_tmp = penlift_kin_aver(:,sub);
    
    EMG_sub = NDD(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            dim, kin_dim);

    fieldname = ['sub' num2str(sub)];
    EMG.(['CC_stroke_' fieldname]) = EMG_sub.CC_stroke;
    EMG.(['CC_penlift_' fieldname]) = EMG_sub.CC_penlift;
    EMG.(['MSE_stroke_' fieldname]) = EMG_sub.MSE_stroke;
    EMG.(['MSE_penlift_' fieldname]) = EMG_sub.MSE_penlift;
    EMG.(['VE_stroke_' fieldname]) = EMG_sub.VE_stroke;
    EMG.(['VE_penlift_' fieldname]) = EMG_sub.VE_penlift;
    EMG.(['pred_stroke_' fieldname]) = EMG_sub.pred_stroke;
    EMG.(['pred_penlift_' fieldname]) = EMG_sub.pred_penlift;
    EMG.(['VE_trial_' fieldname]) = EMG_sub.VE_trial;
    EMG.(['CC_trial_' fieldname]) = EMG_sub.CC_trial;
end


%%
function arr = struct_to_arr(struct,dim,num,col)
    if nargin == 4 && ~isempty(col)
        arr = zeros(96*numel(col), num);
        for i = 1:num
            tmp = struct(i).matrix(:,col);  
            arr(:, i) = tmp(:);  
        end
    else
        arr = zeros(dim, num);
        for i = 1:num
            arr(:, i) = struct(i).matrix(:);    
        end
    end
end

function B = ridge_reg(dataX, dataY,lambda_vals)
    dataX=dataX';
    dataY=dataY';
    B = (dataX' * dataX + lambda_vals * eye(size(dataX, 2))) \ (dataX' * dataY);
end

function [B_dyn, spk_basisFxns] = LDR_train(spk_aver, kin_aver, ind_aver, neural_dim, kin_dim)
    spk_aver(ind_aver) = [];
    kin_aver(ind_aver) = [];

    spk_data = struct();
    for i = 1:numel(spk_aver)
        spk_data(i).matrix = spk_aver{i};
    end

    kin_data = struct();
    for i = 1:numel(kin_aver)
        kin_data(i).matrix = kin_aver{i}(kin_dim,:);
    end
    
    hasNaN = arrayfun(@(x) any(isnan(x.matrix(:))), kin_data);
    kin_data = kin_data(~hasNaN);
    spk_data = spk_data(~hasNaN);

    [spk_basisFxns, spk_act, ~] = eigTransform(spk_data, neural_dim);% extract neural dynamics from training dataset

    kin_act = kin_data;
    for i = 1:numel(kin_data)
        kin_act(i).matrix = kin_data(i).matrix * pinv(spk_basisFxns.');
    end
    
    kin_tr = struct_to_arr(kin_act, numel(kin_dim)*neural_dim, numel(kin_act),[]);
    spk_tr = struct_to_arr(spk_act, 96*neural_dim, numel(spk_act),[]);

    lambda_vals = 1000;
    B_dyn = ridge_reg(spk_tr, kin_tr, lambda_vals);
end

function results = ...
    NDD(stroke, stroke_kin, stroke_ind, stroke_aver, stroke_kin_aver, stroke_ind_aver, ...
                                penlift, penlift_kin, penlift_ind, penlift_aver, penlift_kin_aver, penlift_ind_aver, ...
                                dim, kin_dim)

CC_stroke = zeros(size(stroke_ind,1),1);
MSE_stroke = zeros(size(stroke_ind,1),1);
VE_stroke = zeros(size(stroke_ind,1),1);
pred_stroke = cell(size(stroke_ind,1),1);

CC_penlift = zeros(size(penlift_ind,1),1);
MSE_penlift = zeros(size(penlift_ind,1),1);
VE_penlift = zeros(size(penlift_ind,1),1);
pred_penlift = cell(size(penlift_ind,1),1);

VE_trial = [];
CC_trial = [];
for target = 1:30 % leave-one-character-out
    %% === Stroke ===
    target_ind_s = find(stroke_ind_aver == target);
    [B_stroke, spk_basis_s] = LDR_train(stroke_aver, stroke_kin_aver, target_ind_s, dim, kin_dim);

    target_ind_s = find(stroke_ind == target);
    spk_tt = stroke(target_ind_s);
    kin_tt = stroke_kin(target_ind_s);
    kin_tt = cellfun(@(x) x(kin_dim, :), kin_tt, 'UniformOutput', false);

    spk_act = struct();
    for cond = 1:numel(spk_tt)
        spk_act(cond).matrix = spk_tt{cond} * pinv(spk_basis_s.');
    end

    stroke_spk_tt = struct_to_arr(spk_act, 96 * dim, numel(spk_act), []);

    for i_stroke = 1:size(stroke_spk_tt, 2)
        pred_s = reshape(B_stroke' * stroke_spk_tt(:, i_stroke), numel(kin_dim), dim) * spk_basis_s';
        CC_stroke(target_ind_s(i_stroke)) = getcc(kin_tt{i_stroke}, pred_s);
        MSE_stroke(target_ind_s(i_stroke)) = mean(mean((kin_tt{i_stroke} - pred_s).^2, 2)');
        VE_stroke(target_ind_s(i_stroke)) = getve(kin_tt{i_stroke}, pred_s);
        pred_stroke{target_ind_s(i_stroke)}= pred_s;
    end

    %% === Penlift ===
    target_ind_p = find(penlift_ind_aver == target);
    [B_penlift, spk_basis_p] = LDR_train(penlift_aver, penlift_kin_aver, target_ind_p, dim, kin_dim);

    target_ind_p = find(penlift_ind == target);
    spk_tt = penlift(target_ind_p);
    kin_tt = penlift_kin(target_ind_p);
    kin_tt = cellfun(@(x) x(kin_dim, :), kin_tt, 'UniformOutput', false);

    spk_act = struct();
    for cond = 1:numel(spk_tt)
        spk_act(cond).matrix = spk_tt{cond} * pinv(spk_basis_p.');
    end
    
    penlift_spk_tt = struct_to_arr(spk_act, 96 * dim, numel(spk_act), []);

    for i_penlift = 1:size(penlift_spk_tt, 2)
        pred_p = reshape(B_penlift' * penlift_spk_tt(:, i_penlift), numel(kin_dim), dim) * spk_basis_p';
        CC_penlift(target_ind_p(i_penlift)) = getcc(kin_tt{i_penlift}, pred_p);
        MSE_penlift(target_ind_p(i_penlift)) = mean(mean((kin_tt{i_penlift} - pred_p).^2, 2)');
        VE_penlift(target_ind_p(i_penlift)) = getve(kin_tt{i_penlift}, pred_p);
        pred_penlift{target_ind_p(i_penlift)} = pred_c;
    end
   %% === variance explained of each character ===
   target_ind_s = reshape(target_ind_s,size(target_ind_s,1)/3,3);
   target_ind_p = reshape(target_ind_p,size(target_ind_p,1)/3,3);
   for trial=1:3
       stroke_kin_tmp = stroke_kin(target_ind_s(:,trial));
       penlift_kin_tmp = penlift_kin(target_ind_p(:,trial));
       pred_stroke_tmp = pred_stroke(target_ind_s(:,trial));
       pred_penlift_tmp = pred_penlift(target_ind_p(:,trial));
        
       trial_kin_cell = cell(1, length(stroke_kin_tmp)+length(penlift_kin_tmp));
       pred_trial_cell = cell(1, length(pred_stroke_tmp)+length(pred_penlift_tmp));
       for i=1:length(pred_stroke_tmp)
           trial_kin_cell{i*2-1} = stroke_kin_tmp{i};
           pred_trial_cell{i*2-1} = pred_stroke_tmp{i};
       end
       for i=1:length(pred_penlift_tmp)
           trial_kin_cell{i*2} = penlift_kin_tmp{i};
           pred_trial_cell{i*2} = pred_penlift_tmp{i};
       end
       trial_kin = cell2mat(trial_kin_cell);
       trial_kin = trial_kin(kin_dim, :);

       pred_trial = cell2mat(pred_trial_cell);  % prediction of each character

        VE_trial = [VE_trial; getve(trial_kin, pred_trial)];
        CC_trial = [CC_trial; getcc(trial_kin, pred_trial)];
   end

end
results = struct();
results.CC_stroke   = CC_stroke;
results.MSE_stroke  = MSE_stroke;
results.VE_stroke  = VE_stroke;
results.pred_stroke = pred_stroke;
results.CC_penlift   = CC_penlift;
results.MSE_penlift  = MSE_penlift;
results.VE_penlift  = VE_penlift;
results.pred_penlift = pred_penlift;
results.VE_trial = VE_trial;
results.CC_trial = CC_trial;
end

function cc = getcc(data, data_approx)

    tmp_cc = [];
    for i=1:size(data,1)
        tmp = corrcoef(data(i,:),data_approx(i,:));
        tmp_cc = [tmp_cc tmp(1,2)];
    end
    cc= nanmean(tmp_cc);
end

function VE = getve(data, data_approx)
    VE = 1 -sum(var(data - data_approx,0,2))/ ...
        sum(var(data,0,2));
end
