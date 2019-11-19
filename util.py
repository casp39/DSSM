import torch

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def one_hot_embedding(labels, num_classes):
    '''
    input:
    output: (batch_size, class_size)
    '''
    y = torch.eye(num_classes)
    return y[labels]

def create_cmx(args, np_pred, np_true):
    cmx_data = confusion_matrix(np_true, np_pred).astype('float64')
    for i in range(60):
        cmx_data[i] = cmx_data[i] / np.sum(cmx_data[i])
    df_cmx = pd.DataFrame(cmx_data)
    plt.figure(figsize = (30, 21))
    sns.heatmap(df_cmx.round(2), annot=True, cmap="Blues")
    plt.savefig(args.save_path + '/cmx.png')
