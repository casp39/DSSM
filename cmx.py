import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.switch_backend('agg')

plt.rcParams["font.size"] = 11
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

for i in range(1,5):
    save_path = './log/dssm_view' +str(i) 
    np_pred = np.load(save_path + '/pred.npy')
    np_true = np.load(save_path + '/true.npy')

    class_name = ['drink water', 'brushing teeth', 'clapping', 'reading', 'writing', 'phone call', 'playing with phone',
              'rub two hands', 'headache', 'back pain', 'neck pain', 'fan self']

    cmx_data = confusion_matrix(np_true, np_pred).astype('float64')
    for i in range(60):
        cmx_data[i] = cmx_data[i] / np.sum(cmx_data[i])
    # cmx_data = cmx_data[[9,10,11,28,29,33,36,43,45,46,48], :]
    # cmx_data = cmx_data[:, [9,10,11,28,29,33,36,43,45,46,48]]
    cmx_data = cmx_data[[0,2,9,10,11,28,29,33,43,45,46,48], :]
    cmx_data = cmx_data[:, [0,2,9,10,11,28,29,33,43,45,46,48]]
    df_cmx = pd.DataFrame(cmx_data, index=class_name, columns=class_name)
    plt.figure(figsize = (8, 7))
    sns.heatmap(df_cmx.round(2), annot=True, cmap="Blues", vmax=0.7, vmin=0)
    plt.xticks(rotation=30)
    plt.savefig(save_path + '/cmx_part.pdf', bbox_inches="tight")
