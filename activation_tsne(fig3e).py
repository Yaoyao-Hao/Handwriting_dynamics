import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# TSNE for neural activations (fig.3e)
example_data = '0623_mua_noaver_smo.mat'
data = sio.loadmat(example_data)
stroke = data['Stroke_cell']
penlift = data['Penlift_cell']

example_data = 'acrosscond_LDS_0623.mat'
data = sio.loadmat(example_data)
activation = data['act']['matrix'][0]

activation_plt = []
for i in range(activation.shape[0]):
    mat = activation[i]
    activation_plt.append(mat.flatten(order='F'))

activation_plt = np.array(activation_plt)

# tsne 3d
y = np.concatenate([np.zeros(stroke.shape[0]), np.ones(penlift.shape[0])])
tsne = TSNE(n_components=3, perplexity=5, learning_rate=100, random_state=1111)
X_tsne = tsne.fit_transform((activation_plt - activation_plt.mean(axis=1, keepdims=True)) / activation_plt.std(axis=1, keepdims=True))

# idx = np.argsort(X_tsne[:, 0])[-2:][::-1]
# X_tsne = np.delete(X_tsne,idx, axis=0)  # remove outlier
# y = np.delete(y, idx, axis=0)

fig = plt.figure(figsize=(4.4 / 2.54, 5 / 2.54))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(7,300)

for label, color, name in zip([0, 1], ['#D6251F', '#3778AD'], ['Stroke', 'Pen Lift']):
    ax.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], X_tsne[y == label, 2],
               label=name, c=color, s=0.8, edgecolors='none')

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.line.set_linewidth(0.5)

ax.set_box_aspect([1, 1, 1.04])
ax.set_xlim(-35, 35)
ax.set_ylim(-35, 35)
ax.set_zlim(-30, 30)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)
plt.show()
