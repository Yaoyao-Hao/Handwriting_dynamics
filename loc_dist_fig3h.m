%% offset distance of rotational planes within- and across-conditions  fig3h
clc
clear all

example_data = 'acrosscond_LDS_0623.mat';
data = load(example_data);
act = data.act;

example_data = '0623_mua_noaver_smo.mat';
data = load(example_data);
stroke_ind = data.stroke_ind;
penlift_ind = data.penlift_ind;
% stroke
offset_same_stroke =[];
offset_dist_s = [];
for cond1=1:size(stroke_ind,1)
    for cond2 = 1:size(stroke_ind,1)
        offset_dist_s(cond1,cond2) = norm(act(cond1).matrix(:,1)-act(cond2).matrix(:,1));
    end
end
[offset_same_stroke,offset_dist_s, same_stroke_ind] = dist_in_same(stroke_ind, offset_dist_s); 
% pen lift
offset_same_penlift =[];
offset_dist_p = [];
for cond1=1:size(penlift_ind,1)
    for cond2 = 1:size(penlift_ind,1)
        offset_dist_p(cond1,cond2) = norm(act(size(stroke_ind,1)+cond1).matrix(:,1)-act(size(stroke_ind,1)+cond2).matrix(:,1));
    end
end
[offset_same_penlift,offset_dist_p, same_penlift_ind] = dist_in_same(penlift_ind, offset_dist_p);

% stroke
offset_dist_s = offset_dist_s(~isnan(offset_dist_s));
offset_dist_s = offset_dist_s(offset_dist_s ~= 0);
p = ranksum(offset_dist_s, offset_same_stroke)

% normalization
min_val = min([offset_dist_s(:); offset_same_stroke(:)]);
max_val = max([offset_dist_s(:); offset_same_stroke(:)]);
offset_dist_s = (offset_dist_s - min_val) / (max_val - min_val);
offset_same_stroke = (offset_same_stroke - min_val) / (max_val - min_val);

% penlift
offset_dist_p = offset_dist_p(~isnan(offset_dist_p));
offset_dist_p = offset_dist_p(offset_dist_p ~= 0);
p = ranksum(offset_dist_p, offset_same_penlift)

% normalization
min_val = min([offset_dist_p(:); offset_same_penlift(:)]);
max_val = max([offset_dist_p(:); offset_same_penlift(:)]);
offset_dist_p = (offset_dist_p - min_val) / (max_val - min_val);
offset_same_penlift = (offset_same_penlift - min_val) / (max_val - min_val);

%% function

function [dist_same,dist_stroke,ind] = dist_in_same(stroke_ind, dist_stroke)
    ind = [];
    for charac=1:30
        charac_ind = find(stroke_ind==charac);
        charac_ind = reshape(charac_ind,size(charac_ind,1)/3,3); % stroke num* trials
        ind = [ind;charac_ind];
    end
    
    dist_same = [];
    for i_stroke=1:size(ind,1)
        all_perms = perms(ind(i_stroke,:));
        stroke_all_ind = unique(all_perms(:,1:2), 'rows'); 
    
        [~,row_ind,~] =  unique(sort(stroke_all_ind, 2), 'rows', 'stable');
        stroke_unique_ind = stroke_all_ind(row_ind, :);
        
        for i_ind = 1:size(stroke_unique_ind,1)
            dist_same = [dist_same dist_stroke(stroke_unique_ind(i_ind,1), stroke_unique_ind(i_ind,2))]; 
        end

        for i_ind = 1:size(stroke_all_ind,1)
            dist_stroke(stroke_all_ind(i_ind,1), stroke_all_ind(i_ind,2)) = nan;
        end
        
    end

end