%% alignment index of rotational planes within- and across-conditions fig3i
clc
clear all
currpath = pwd;
% addpath([currpath '/utils/TBFandCondSpecificDynamics']);
% addpath([currpath '/utils/evalFit']);
% addpath([currpath '/TBF']);

dimUse=5;

example_data = 'acrosscond_LDS_0623.mat';
data = load(example_data);
basisFxns = data.basisFxns;
act = data.act;
stroke_approx = data.stroke_approx;
penlift_approx = data.penlift_approx;

example_data = '0623_mua_noaver_smo.mat';
data = load(example_data);
stroke_ind = data.stroke_ind;
penlift_ind = data.cohesion_ind;

example_data = '0623_mua_noaver_smo.mat';
data = load(example_data);
stroke = data.Stroke_cell;
penlift = data.Penlift_cell;

aligned_stroke = nan(size(stroke_approx,2),size(stroke_approx,2),(dimUse-1)/2);  % cond x cond x plane
for plane=1:(dimUse-1)/2
    inds = plane*2:plane*2+1;
    for cond1=1:size(stroke_approx,2)
        for cond2 = 1:size(stroke_approx,2)
            aligned_stroke(cond1,cond2,plane) = getAlignmentIndex(stroke{cond1},...
                act(cond2).matrix(:,inds)); 
        end
    end
end

aligned_penlift = zeros(size(penlift_approx,2),size(penlift_approx,2),(dimUse-1)/2);  % cond x cond x plane
for plane=1:(dimUse-1)/2
    inds = plane*2:plane*2+1;
    for cond1=1:size(penlift_approx,2)
        for cond2 = 1:size(penlift_approx,2)
            aligned_penlift(cond1,cond2,plane) = getAlignmentIndex(penlift{cond1}, ...
                act(cond2+size(stroke_approx,2)).matrix(:,inds));
        end
    end
end

% within-stroke or penlift
[aligned_same_stroke,aligned_stroke, same_stroke_ind]= aligned_in_same(stroke_ind, aligned_stroke, dimUse);
[aligned_same_penlift,aligned_penlift, same_penlift_ind] = aligned_in_same(penlift_ind, aligned_penlift, dimUse);

aligned_stroke_tmp = [];
aligned_penlift_tmp = [];
for plane_plot=1:2
    aligned_stroke_plane = aligned_stroke(:,:,plane_plot);
    aligned_stroke_plane = aligned_stroke_plane(~isnan(aligned_stroke_plane));
    aligned_stroke_plane = aligned_stroke_plane(aligned_stroke_plane ~= 0);
    aligned_stroke_tmp = [aligned_stroke_tmp aligned_stroke_plane];

    aligned_penlift_plane = aligned_penlift(:,:,plane_plot);
    aligned_penlift_plane = aligned_penlift_plane(~isnan(aligned_penlift_plane));
    aligned_penlift_plane = aligned_penlift_plane(aligned_penlift_plane ~= 0);
    aligned_penlift_tmp = [aligned_cohesion_tmp aligned_penlift_plane];
end   
% normalize
min_val = min([aligned_stroke_tmp(:); aligned_same_stroke(:)]);
max_val = max([aligned_stroke_tmp(:); aligned_same_stroke(:)]);
aligned_stroke = (aligned_stroke_tmp - min_val) / (max_val - min_val);
aligned_same_stroke = (aligned_same_stroke - min_val) / (max_val - min_val);

min_val = min([aligned_penlift_tmp(:); aligned_same_penlift(:)]);
max_val = max([aligned_penlift_tmp(:); aligned_same_penlift(:)]);
aligned_penlift = (aligned_penlift_tmp - min_val) / (max_val - min_val);
aligned_same_penlift = (aligned_same_penlift - min_val) / (max_val - min_val);


%% function
function [aligned_same,aligned_stroke,ind] = aligned_in_same(stroke_ind, aligned_stroke, dimUse)
    ind = [];
    for charac=1:30
        charac_ind = find(stroke_ind==charac);
        charac_ind = reshape(charac_ind,size(charac_ind,1)/3,3);
        ind = [ind;charac_ind];
    end
    
    aligned_same = [];
    for plane=1:(dimUse-1)/2
        aligned_same_plane = [];
        for i_stroke=1:size(ind,1)
            all_perms = perms(ind(i_stroke,:));
            stroke_all_ind = unique(all_perms(:,1:2), 'rows');

            [~,row_ind,~] =  unique(sort(stroke_all_ind, 2), 'rows', 'stable');
            stroke_unique_ind = stroke_all_ind(row_ind, :);
            
            for i_ind = 1:size(stroke_unique_ind,1)
                aligned_same_plane = [aligned_same_plane aligned_stroke(stroke_unique_ind(i_ind,1), stroke_unique_ind(i_ind,2),plane)];
            end

            for i_ind = 1:size(stroke_all_ind,1)
                aligned_stroke(stroke_all_ind(i_ind,1), stroke_all_ind(i_ind,2),plane) = nan;
            end
            
        end

        aligned_same = [aligned_same;aligned_same_plane];
    end

end