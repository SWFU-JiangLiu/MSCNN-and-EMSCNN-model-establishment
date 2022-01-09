# coding=utf-8
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2 as cv
import os
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.vis_utils import plot_model
def cv_imread(filePath):
    cv_img = cv.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img
def get_data(data_dir, labels, img_size):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        path = path + '/'
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv_imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                resized_arr = cv.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
def train_test_data(train, val,img_size):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)
    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)
    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255
    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)
    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)
    return x_train, y_train, x_val, y_val
# plot loss
def loss_plot(hist, title,model_name):
    plt.figure(dpi=600)
    plt.plot(np.arange(len(hist.history['loss'])), hist.history['loss'], label='training')
    plt.plot(np.arange(len(hist.history['val_loss'])), hist.history['val_loss'], label='validation')
    plt.title(title + ' Training and Validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=0)
    img_title =  './Result/'+model_name+title + '_loss.png'
    plt.savefig(img_title)
    plt.show()
    return 0
# plot acc
def acc_plot(hist, title, flag,model_name):
    plt.figure(dpi=600)
    val_flag = 'val_' + flag
    plt.plot(np.arange(len(hist.history[flag])), hist.history[flag], label='training')
    plt.plot(np.arange(len(hist.history[val_flag])), hist.history[val_flag], label='validation')
    plt.title(title + ' Training and Validation')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc=0)
    img_title = './Result/'+ model_name+title + '.png'
    plt.savefig(img_title)
    plt.show()
    return 0
def plot_confusion_matrix(confusion_mat, title, target_names):
    save_imgname = './Result/'+title + '.png'
    sns.set()
    fig, ax = plt.subplots(figsize=(16,10))
    label_txt =  target_names
    con_mat_norm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    con_mat_norm = np.around(con_mat_norm, decimals=2)
    sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
    ax.set_title(title)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    ax.set_xticklabels(label_txt, rotation=50, horizontalalignment='left', family='Times New Roman', fontsize=8)
    ax.set_yticklabels(label_txt, rotation=0, family='Times New Roman', fontsize=8)
    ax.xaxis.set_ticks_position("top")
    fig.tight_layout()
    fig.savefig(save_imgname, dpi=600)
    plt.show()
def evalIndex(model, x_val, y_val, target_name, title, model_title):
    pred_savepath = model_title + '_predict.csv'
    pred_argmax_savepath = model_title + '_predict_argmax.csv'
    pred_conf_savepath = model_title + '_con_matrix.csv'
    true_y_savepath = model_title+'_true_lab.csv'
    true_lab = y_val
    np.savetxt(true_y_savepath, true_lab)
    pred_lab = model.predict(x_val)
    np.savetxt(pred_savepath, pred_lab)
    pred_lab = np.argmax(pred_lab, axis=1)
    np.savetxt(pred_argmax_savepath, pred_lab)
    pred_lab = pred_lab.reshape(1, -1)[0]
    conf_mat = confusion_matrix(y_true=true_lab, y_pred=pred_lab)
    titles = 'Confusion matrix of ' + title
    plot_confusion_matrix(conf_mat, titles, target_name)
    print('Confusion matrix:\n', conf_mat)
    np.savetxt(pred_conf_savepath, conf_mat)
    print('classification_report:\n', classification_report(y_val, pred_lab,
                                                            target_names=target_name))
    print('acc:\n', accuracy_score(pred_lab, true_lab))
def cnn_model(x_train, y_train, x_val, y_val,epoch,dense,nb_class,chidu,model_name,img_size):
    model = Sequential()
    model.add(Conv2D(64, chidu, padding="same", activation="relu", input_shape=(img_size, img_size, 3)))
    model.add(MaxPool2D())
    model.add(Conv2D(64,chidu, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(32, chidu, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(32, chidu, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(dense, activation="relu"))
    model.add(Dense(nb_class, activation="softmax"))
    model.summary()
    to_file='./Result/'+model_name+'model_plot.png'
    plot_model(model, to_file=to_file, show_shapes=True, show_layer_names=True,dpi=600)
    opt = 'rmsprop'
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epoch, validation_data=(x_val, y_val))
    return model, history
if __name__ == '__main__':
    # 1、load data
    # ----------------------------------------------------------------------------------
    path = r"./data/"
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
              '11', '12', '13', '14', '15', '16', '17', '18', '19',
              '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    lab_id = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
              '11', '12', '13', '14', '15', '16', '17', '18', '19',
              '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    # 2
    # ----------------------------------------------------------------------------------
    img_size = 112
    dense = 500
    epoch = 30
    nb_class = 30
    chidustr=['(5,5)','(2,2)','(2,3)','(3,2)','(3,3)']
    chidus=[(5,5),(2,2),(2,3),(3,2),(3,3)]
    time=[]
    # 3 load train and val
    # ----------------------------------------------------------------------------------
    train_path = path + '/train'
    test_path = path + '/val'
    train = get_data(train_path, labels, img_size)
    val = get_data(test_path, labels, img_size)
    # 4 datagen
    # ----------------------------------------------------------------------------------
    x_train, y_train, x_val, y_val = train_test_data(train, val, img_size)
    train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
    val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
    train_generator.fit(x_train)
    val_generator.fit(x_val)
    # 5、train ['(5,5)','(2,2)','(2,3)','(3,2)','(3,3)']
    # ----------------------------------------------------------------------------------
    for n in range(5):
        starttime = datetime.datetime.now()
        chidu =chidustr[n]
        model_name = 'model' + str(chidu)
        x1=chidu.split(',')[0].split('(')[1]
        x2=chidu.split(',')[1].split(')')[0]
        title = 'Bird classification based on CNN(Scale'+x1+'x'+x2+')'
        # 6 model training
        # ----------------------------------------------------------------------------------
        model, history = cnn_model(x_train, y_train, x_val, y_val,epoch,dense,nb_class,chidus[n],model_name,img_size)
        # 7 Model evaluation
        # ----------------------------------------------------------------------------------
        acc_plot(history, title, 'accuracy',model_name)
        loss_plot(history, title,model_name)
        model_title='./Result/'+model_name
        evalIndex(model, x_val, y_val,lab_id,title,model_title)
        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds)
        time.append((endtime - starttime).seconds)
    print('model(5,5)(2,2)(2,3)(3,2)(3,3) train time:')
    print(time)
