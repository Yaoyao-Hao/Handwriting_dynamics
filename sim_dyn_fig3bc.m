%% simulated dynamics fig.3bc
clc
clear all
% addpath([currpath '/utils/TBFandCondSpecificDynamics']);
% addpath([currpath '/utils/evalFit']);
% addpath([currpath '/TBF']);

example_data = '0623_mua_noaver_smo.mat';
data = load(example_data);
stroke = data.Stroke_cell;
penlift = data.Penlift_cell;

r = 0.01:0.05:0.99;
i = 0.01:0.05:0.99;

eigVals_cell = cell(80000,1);
dynamics_cell = cell(80000,1);

VE_stroke = cell(size(dynamics_cell,1),1);
VE_penlift = cell(size(dynamics_cell,1),1);

idx = 1;
for r1 = r
    for i1 = i % eigenvalues of rotation1
        for r2 = r
            for i2 = i % eigenvalues of rotation2
                if atan2(i1,r1)<atan2(i2,r2)
                    disp(['r1= ' num2str(r1) ', i1= ' num2str(i1) ',r2= ' num2str(r2) ', i2= ' num2str(i2)]) 
                    eigVals = [1; complex(r1,i1); complex(r1,-i1); complex(r2,i2); complex(r2,-i2)];% eigenvalues
                    sim_dynamics = simulated_dyns(eigVals);
                    
                    [~, VE_tmp] = reconstruct(sim_dynamics,stroke,1:96);
                    VE_stroke{idx} = VE_tmp.mean;
                    [~, VE_tmp] = reconstruct(sim_dynamics,penlift,1:96);
                    VE_penlift{idx} = VE_tmp.mean;

                    eigVals_cell{idx} = eigVals;
                    dynamics_cell{idx} = sim_dynamics;
                    idx = idx+1;
                end
            end
        end
    end
end
% variance explained of neural activities using sim. dyn.
VE_s = cell2mat(VE_s);
VE_s = VE_s(:);
VE_c = cell2mat(VE_c);
VE_c = VE_c(:);