# coding=utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.algorithms import mode
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
def plot_confusion_matrix(confusion_mat, title, target_names):
    save_imgname = title + '.png'
    sns.set()
    fig, ax = plt.subplots(figsize=(12,8))
    label_txt =  target_names
    con_mat_norm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)
    sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
    ax.set_title(title)  # 标题
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    ax.set_xticklabels(label_txt, rotation=50, horizontalalignment='left', family='Times New Roman', fontsize=10)
    ax.set_yticklabels(label_txt, rotation=0, family='Times New Roman', fontsize=10)
    ax.xaxis.set_ticks_position("top")
    fig.tight_layout()
    fig.savefig(save_imgname, dpi=600)
    plt.show()
if __name__ == '__main__':

    lab_id = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
              '11', '12', '13', '14', '15', '16', '17', '18', '19',
              '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    df=pd.read_csv(r'./data/EMSCNN_train.csv')
    true_lab=df['TRUE']
    # ------------------------------
    pre1=df['WT2X2']
    pre2=df['WT2X3']
    pre3=df['WT3X2']
    pre4=df['WT3X3']
    pred_lab = np.array([])
    for i in range(0,len(true_lab)):
        pred_lab = np.append(pred_lab, mode([pre1[i],pre2[i],pre3[i],pre4[i]])[0])
    print(pred_lab)
    np.savetxt('Ensemble_data/22-23-32-33_WT.csv',pred_lab)
    print(pred_lab.shape)
    pred_lab = pred_lab.reshape(1, -1)[0]
    print(pred_lab.shape)
    conf_mat = confusion_matrix(y_true=true_lab, y_pred=pred_lab)
    plot_confusion_matrix(conf_mat, 'Confusion matrix of bird classification based on WT and EMSCNN',lab_id)
    print('Confusion matrix:\n', conf_mat)
    print('classification_report:\n', classification_report(true_lab, pred_lab,
                                    target_names=lab_id))
    print('acc:\n', accuracy_score(pred_lab, true_lab))