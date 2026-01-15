# Handwriting_dynamics
This repository contains the core codes for the paper Wang et al., 2025 "Neural dynamics underlying complex handwriting movements".

## Installation
To run the Python scripts in this repository, Python packages in the `requirements.txt` need to be installed with `Python==3.9`:
```
pip install -r requirements.txt
```

## Usage
* **Orth_subspace_fig2d.m**, **Excl_shared_subspace_fig2fh.m:**
These two modules estimate subspaces on the Stiefel manifold using the manopt toolbox for MATLAB.

Orthogonal subspaces for strokes and pen lifts:
```MATLAB
[Q, ~, info, options] = orthogonal_subspaces(C_stroke,d_st,C_penlift,d_pl); 
```
Exclusive subspaces for strokes and pen lifts:
```MATLAB
[Q_st,flagSt] = exclusive_subspace(C_stroke,C_penlift,d_st,alphaNullSpace);
```
```MATLAB
[Q_pl,flagPl] = exclusive_subspace(C_penlift,C_stroke,d_pl,alphaNullSpace);
```
Shared subspace for strokes and pen lifts:
```MATLAB
[Q1,Qshared, Qcost, info, options] = shared_subspace(Q_st,d_st,Q_pl,d_pl,C_stroke,C_penlift,d_shared);
```
* **shared_neural_dynamics_fig3a.m:**
This module uses unconstrained methods to extract shared neural dynamics of strokes and pen lifts:
```MATLAB
[basisFxns,loadings,params] = eigTransform([stroke_data,penlift_data],dimUse);
```
The extracted dynamics are then used to reconstruct the neural signals via a least-squares approach:
```MATLAB
act_stroke = struct();
for cond = 1:size(stroke,1)
    act_stroke(cond).matrix = stroke_data(cond).matrix*pinv(basisFxns.');
end
stroke_approx = struct();
for cond = 1:size(act_stroke,2)
    stroke_approx(cond).matrix = act_stroke(cond).matrix*basisFxns';
end
```
* **activation_tsne(fig3e).py**, **loc_tsne(fig3g).py:**
These two modules apply t-SNE for dimensionality reduction and visualization of neural activations and state space location of neural dynamics.

t-SNE for neural activations:
```Python
tsne = TSNE(n_components=3, perplexity=5, learning_rate=100, random_state=1111)
X_tsne = tsne.fit_transform((activation_plt - activation_plt.mean(axis=1, keepdims=True)) / activation_plt.std(axis=1, keepdims=True))
```
t-SNE for state space location of neural dynamics:
```Python
tsne = TSNE(n_components=3, perplexity=5, learning_rate=100, random_state=42)
X_tsne = tsne.fit_transform((activation_plt - activation_plt.mean(axis=1, keepdims=True)) / activation_plt.std(axis=1, keepdims=True))
```
* **loc_dist_fig3h.m:**
This module compares the state space location of rotational planes within- and across-conditions:
```MATLAB
offset_dist_s(cond1,cond2) = norm(act(cond1).matrix(:,1)-act(cond2).matrix(:,1));
offset_dist_p(cond1,cond2) = norm(act(size(stroke_ind,1)+cond1).matrix(:,1)-act(size(stroke_ind,1)+cond2).matrix(:,1));
```
* **align_index_fig3i.m:**
This module compares alignment index of rotational planes within- and across-conditions:
```MATLAB
aligned_stroke(cond1,cond2,plane) = getAlignmentIndex(stroke{cond1},act(cond2).matrix(:,inds));
aligned_penlift(cond1,cond2,plane) = getAlignmentIndex(penlift{cond1},  act(cond2+size(stroke_approx,2)).matrix(:,inds));
```
* **Traditional_decoding_fig4e.m:**
This module uses ridge regression to decode multidimensional handwriting from neural signals:
```MATLAB
Vxy = Ridge_decode(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, kin_dim);
```
* **LDR_decoding_fig4e.m:**
This module uses LDR to decode multidimensional handwriting from neural signals:
```MATLAB
Vxy = LDR_decode(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            dim, kin_dim,kin_basis_num);
```
* **NDD_fig4e.m:**
This module uses neural dynamics decoding to decode multidimensional handwriting from neural signals:
```MATLAB
Vxy = NDD(stroke, stroke_kin_tmp, stroke_ind, stroke_aver, stroke_kin_aver_tmp, stroke_ind_aver, ...
                            penlift, penlift_kin_tmp, penlift_ind, penlift_aver, penlift_kin_aver_tmp, penlift_ind_aver, ...
                            dim, kin_dim);
```
* **crossday_decoding_rawsignal_fig5cde.m:**
This module perform cross-day decoding using raw neural signals:
```MATLAB
model = fitcsvm(data_tr, tr_labels, 'KernelFunction', 'linear'); % training a svm
y_pred = predict(model, data_tt);
```
* **crossday_decoding_activations_fig5cde.m:**
This module perform cross-day decoding using neural activations. Firstly, extracting neural dynamics from training sessions and training a svm using neural activations:
```MATLAB
[basisFxns,act,params] = eigTransform([stroke_data_tr,penlift_data_tr],dimUse);
```
```MATLAB
data_tr = [];
for i = 1:length(act)
    data_tr = [data_tr; act(i).matrix(:)'];
end
model = fitcsvm(data_tr, tr_labels, 'KernelFunction', 'linear'); % training a svm
```
Then, performing cross-day decoding using neural activations from test sessions
```MATLAB
y_pred = predict(model, act');
```
## Acknowledgements
We thank the authors of the following open-source projects for making their code publicly available:
- Jonathan et al., subspace-opt, [link](https://github.com/jcykao/subspace-opt)
- Sabatini and Kaufman, LDR-public, [link](https://github.com/kaufmanlab/LDR-public)
