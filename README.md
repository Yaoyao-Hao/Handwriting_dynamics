# Handwriting_dynamics
This repository contains the core codes for the paper Wang et al., 2025 "Neural dynamics underlying complex handwriting movements".

## Installation
To run the Python scripts in this repository, Python packages in the `requirements.txt` need to be installed with `Python==3.9`:
```
pip install -r requirements.txt
```

## Usage
* **Orth_subspace_fig2d.m**, **Excl_shared_subspace_fig2fh.m**
These two modules estimate subspaces on the Stiefel manifold using the manopt toolbox for MATLAB.
Orthogonal subspaces for strokes and pen lifts:
```MATLAB
[Q, ~, info, options] = orthogonal_subspaces(C_stroke,d_st,C_penlift,d_pl); 
```
Exclusive subspaces for strokes and pen lifts:
```MATLAB
[Q_st,flagSt] = exclusive_subspace(C_stroke,C_penlift,d_st,alphaNullSpace);
[Q_pl,flagPl] = exclusive_subspace(C_penlift,C_stroke,d_pl,alphaNullSpace);
```
Shared subspace for strokes and pen lifts:
```MATLAB
[Q1,Qshared, Qcost, info, options] = shared_subspace(Q_st,d_st,Q_pl,d_pl,C_stroke,C_penlift,d_shared);
```
