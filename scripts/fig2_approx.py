from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


f = np.load('./outdir/loss_diffs_small_mnist_all_cnn_c_99999_retrain-100-finetune-all.npz')

actual_loss_diffs = f['actual_loss_diffs']
predicted_loss_diffs = f['predicted_loss_diffs']

# for i in range(0, len(predicted_loss_diffs)):
#     predicted_loss_diffs[i] = predicted_loss_diffs[i]*1000


sns.set_style('white')
fontsize=16
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(4, 5))

ax.set_aspect('equal')
ax.set_xlabel('Actual diff in loss', fontsize=fontsize)

ax.set_xticks(np.arange(-0.06, 0.06, 0.03))
ax.set_yticks(np.arange(-0.06, 0.06, 0.03))
ax.set_xlim([-0.05, 0.05])
ax.set_ylim([-0.05, 0.05])
ax.plot([-0.05, 0.05], [-0.05, 0.05], 'k-', alpha=0.2, zorder=1)
ax.plot([0.00, 0.00], [-0.05, 0.05], 'k-', alpha=0.1, zorder=1)
ax.plot([-0.05, 0.05], [0.00, 0.00],'k-', alpha=0.1, zorder=1)
    
ax.set_ylabel('Predicted diff in loss', fontsize=fontsize)

ax.scatter(actual_loss_diffs, predicted_loss_diffs, zorder=2)
ax.set_title('torch: CNN softmax(approx)-all', fontsize=fontsize)

plt.savefig('fig_results_small_mnist_all_cnn_c_99999_retrain-100-finetune-all.png')
