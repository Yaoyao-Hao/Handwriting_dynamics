%% origin decoding 
clc
clear all

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
Vxy = Ridge_decode(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            kin_dim);

 % Vz
kin_dim = 3;
Vz = Ridge_decode(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            kin_dim);

% Grip
kin_dim = 4;
Grip = Ridge_decode(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            kin_dim);

% pressure
kin_dim = 5;
Pressure = Ridge_decode(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            kin_dim);

% EMG 6 subjects
EMG = struct();
kin_dim = 6:13;
for sub=1:6
    stroke_kin_tmp = stroke_kin(:,sub);
    penlift_kin_tmp = penlift_kin(:,sub);
    stroke_kin_aver_tmp = stroke_kin_aver(:,sub);
    penlift_kin_aver_tmp = penlift_kin_aver(:,sub);
    
    EMG_sub = Ridge_decode(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            kin_dim);

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

%% function
function concatenatedData = concatenate(structData)
    concatenatedData = [];
    for i = 1:length(structData)
        fieldData = structData(i).matrix;
        concatenatedData = [concatenatedData fieldData];
    end
end

function B_dyn = ridge_train(spk_aver, kin_aver, ind_aver, kin_dim)
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

    kin_tr = concatenate(kin_data)';
    spk_tr = concatenate(spk_data)';

    lambda_vals = 1000;
    B_dyn = (spk_tr' * spk_tr + lambda_vals * eye(size(spk_tr, 2))) \ (spk_tr' * kin_tr);
end


function results = Ridge_decode(stroke, stroke_kin, stroke_ind, stroke_aver, stroke_kin_aver, stroke_ind_aver, ...
                                penlift, penlift_kin, penlift_ind, penlift_aver, penlift_kin_aver, penlift_ind_aver, ...
                                kin_dim)

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
for target = 1:30  % leave-one-character-out
    %% === Stroke ===
    target_ind_s = find(stroke_ind_aver == target);
    B_stroke = ridge_train(stroke_aver, stroke_kin_aver, target_ind_s, kin_dim);

    target_ind_s = find(stroke_ind == target);
    spk_tt = stroke(target_ind_s);
    kin_tt = stroke_kin(target_ind_s);
    kin_tt = cellfun(@(x) x(kin_dim, :), kin_tt, 'UniformOutput', false);

    for i_stroke = 1:size(spk_tt, 1)
        pred_s = B_stroke' * spk_tt{i_stroke};
        CC_stroke(target_ind_s(i_stroke)) = getcc(kin_tt{i_stroke}, pred_s);
        MSE_stroke(target_ind_s(i_stroke)) = mean(mean((kin_tt{i_stroke} - pred_s).^2, 2)');
        VE_stroke(target_ind_s(i_stroke)) = getve(kin_tt{i_stroke}, pred_s);
        pred_stroke{target_ind_s(i_stroke)}= pred_s;
    end

    %% === Penlift ===
    target_ind_p = find(penlift_ind_aver == target);
    B_penlift = ridge_train(penlift_aver, penlift_kin_aver, target_ind_p, kin_dim);

    target_ind_p = find(penlift_ind == target);
    spk_tt = penlift(target_ind_p);
    kin_tt = penlift_kin(target_ind_p);
    kin_tt = cellfun(@(x) x(kin_dim, :), kin_tt, 'UniformOutput', false);
      
    for i_penlift = 1:size(spk_tt, 1)
        pred_p = B_penlift' * spk_tt{i_penlift};
        CC_penlift(target_ind_p(i_penlift)) = getcc(kin_tt{i_penlift}, pred_p);
        MSE_penlift(target_ind_p(i_penlift)) = mean(mean((kin_tt{i_penlift} - pred_p).^2, 2)');
        VE_penlift(target_ind_p(i_penlift)) = getve(kin_tt{i_penlift}, pred_p);
        pred_penlift{target_ind_p(i_penlift)} = pred_p;
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
