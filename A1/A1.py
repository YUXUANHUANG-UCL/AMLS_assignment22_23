import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.models import load_model

def get_data_label(name, google):
    # This function is to get data and corresponding labels
    # get datasets directory
    if google:
        cur_dir = os.path.join('/content/drive/MyDrive/AMLS_assignment22_23/Datasets', name)
    else:
        cur_dir = os.path.join(os.getcwd(), 'Datasets/' + name)
    # get file name and its last name (.jpg or .png)
    lsdir = os.listdir(cur_dir + '/img')
    last_name = '.' + lsdir[0].split('.')[1]
    # get image, using img_1d and img to store 1D image sequences and greyscale 2D image
    img_1d = []
    img = []
    for i in range(len(lsdir)):
        img_1d.append(np.array(Image.open(cur_dir + '/img/' + str(i) + last_name)
                          .convert('L').resize((64, 64))).flatten())
        img.append(np.array(Image.open(cur_dir + '/img/' + str(i) + last_name)
                              .convert('L').resize((64, 64))))
    # load data label file and seperate labels of Task A1 from it
    labels = pd.read_csv(os.path.join(cur_dir, 'labels.csv'))
    labels_first = []
    for i in range(len(labels)):
        label = labels[labels.columns[0]][i].split('\t')
        labels_first.append(label[2])
    return np.array(img_1d), np.array(labels_first), img

def KNNClassifier(X_train, y_train, X_test, k):
    # This function is to construct KNN model and get the prediction results
    # Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    # Fit KNN model
    neigh.fit(X_train, y_train) 
    # get the prediction result of KNN under the value of k
    Y_pred = neigh.predict(X_test)
    return Y_pred

def img_SVM(training_images, training_labels, test_images, test_labels, titles, C, task):
    # This function is to construct SVMs model and get the prediction results
    # define model kernels and pack them together after fitting
    models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(training_images, training_labels) for clf in models)
    pred = []
    # get the results of svm models that use different kernels
    for model, i, title in zip(models, range(2), titles):
        pred.append(model.predict(test_images))
        print(task + ': ' + title + ' ' + "Accuracy:", accuracy_score(test_labels, pred[i]))
    return pred

def get_A1_results():
    # This function is the main function to solve Task A1
    # Get dataset names
    datasets = ['celeba', 'celeba_test', 'cartoon_set', 'cartoon_set_test']
    # whether use google_drive, if use, google_drive = True, if not, google_drive = False
    google_drive = False
    # X_cel is 1D image sequences, y_cel_A1 is the labels of Task A1, X_cel_fig is 2D greyscale images
    X_cel, y_cel_A1, X_cel_fig = get_data_label(datasets[0], google_drive)
    # X_cel_test, y_cel_test_A1, X_cel_test_fig are corresponding test set
    X_cel_test, y_cel_test_A1, X_cel_test_fig = get_data_label(datasets[1], google_drive)

    # Task A1 KNN
    # document knn accuracy
    knn_accuracy = list()
    # document the value of k under the maximum of accuracy
    max_knn = [0, 0]#max_knn = [value, index]
    # find the maximum accuracy of KNN from 1 to 15 of the value of k
    for i in range(1, 16):
        y_pred_A1 = KNNClassifier(X_cel, y_cel_A1, X_cel_test, i)
        knn_accuracy.append(float(np.sum(y_pred_A1 == y_cel_test_A1)) / len(y_pred_A1))
        if max_knn[0] < knn_accuracy[-1]:
            max_knn[0] = knn_accuracy[-1]
            max_knn[1] = i
    # Visualise the results of KNN
    if os.path.exists(os.path.join(os.getcwd(), 'A1')):
        plt.figure()
        plt.plot(range(1,16), knn_accuracy, label='accuracy')
        plt.scatter(max_knn[1], max_knn[0], color='r', label='best point')
        plt.title('The accuracy of KNNs')
        plt.xlabel('value of k')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), os.path.join('A1', 'A1_knn.jpg')))
        plt.close()
    # Using the best value of k to output its accuracy
    y_pred_A1 = KNNClassifier(X_cel, y_cel_A1, X_cel_test, max_knn[1])
    score_A1 = metrics.accuracy_score(y_cel_test_A1, y_pred_A1)
    print(score_A1)
    num_correct = np.sum(y_pred_A1 == y_cel_test_A1)
    accuracy = float(num_correct) / len(y_pred_A1)
    print('Task A1 Got %d / %d correct => accuracy: %f (k = %d)' % (num_correct, len(y_pred_A1), accuracy, max_knn[1]))

    # Task A1 Random Forest
    # document rf accuracy
    rf_accuracy = list()
    # document the number of weak learners under the maximum accuracy of rf model
    max_rf = [0, 0]#max_rf = [value, index]
    # find the maximum accuracy of rf from 10 to 120 of the number of weak learners, the step is 10
    for i in range(10,130,10):
        clf_temp = RandomForestClassifier(n_estimators=i)
        clf_temp.fit(X_cel, y_cel_A1)
        rf_accuracy.append(accuracy_score(y_cel_test_A1, clf_temp.predict(X_cel_test)))
        if max_rf[0] < rf_accuracy[-1]:
            max_rf[0] = rf_accuracy[-1]
            max_rf[1] = i
    # Visualise the results of rf
    if os.path.exists(os.path.join(os.getcwd(), 'A1')):
        plt.figure()
        plt.plot(range(10,130,10), rf_accuracy, label='accuracy')
        plt.scatter(max_rf[1], max_rf[0], color='r', label='best point')
        plt.title('The accuracy of Random Forests')
        plt.xlabel('number of estimators')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), os.path.join('A1', 'A1_rf.jpg')))
        plt.close()

    # create the model - Random Forests
    clf_A1  = RandomForestClassifier(n_estimators=max_rf[1])
    # fit the training data to the model
    clf_A1.fit(X_cel, y_cel_A1)
    # get the prediction of the results under the best number of weak learners
    # (retraining the rf, the best value may change)
    clf_pred_A1 = clf_A1.predict(X_cel_test)
    print("Task A1 Random Forest test Accuracy: %f (number of weak learners = %d)"% (accuracy_score(y_cel_test_A1, clf_pred_A1), max_rf[1]))

    # Task A1 SVMs
    # sklearn functions implementation
    # define the SVM kernels
    titles = ('SVC with linear kernel',
            'SVC with polynomial (degree 3) kernel')
    # get the result of different svm models
    pred_cel_A1 = img_SVM(X_cel, y_cel_A1, X_cel_test, y_cel_test_A1, titles, C=1, task = 'Task_A1')
    
    # Task A1 CNN
    # Split dataset: train, val, test
    X_train_A_fig = tf.stack([(i/255).astype("float64") for i in X_cel_fig[:4500]])
    X_val_A_fig = tf.stack([(i/255).astype("float64") for i in X_cel_fig[4500:5000]])
    X_test_A_fig = tf.stack([(i/255).astype("float64") for i in X_cel_test_fig])
    # split labels and preprocess them
    y_train_A1 = [i == '1' for i in y_cel_A1[:4500]]
    y_val_A1 = [i == '1' for i in y_cel_A1[4500:5000]]
    y_test_A1 = [i == '1' for i in y_cel_test_A1]
    y_train_A1 = tf.stack(y_train_A1)
    y_val_A1 = tf.stack(y_val_A1)
    y_test_A1 = tf.stack(y_test_A1)
    # construct cnn model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.summary()
    # add callbacks to optimise the training process
    # monitor loss to save the best model
    # set earlystopping monitoring accuracy and its patience is 2
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.A1.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)
    ]
    BATCH_SIZE = 500
    EPOCHS = 25
    # configure the model for training
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    # fit the model and save the best model in the same time
    history_A1_2d = model.fit(X_train_A_fig, y_train_A1,
                            epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(X_val_A_fig, y_val_A1),
                            callbacks=callbacks_list)
    # load the best model to do classification and output the results
    model_best = load_model('best_model.A1.h5')
    print('The accuracy of Task A1 by CNN is %.3f' % model_best.evaluate(X_test_A_fig, y_test_A1, verbose=0)[1])
    # visualise the training process and save it to local
    if os.path.exists(os.path.join(os.getcwd(), 'A1')):
        acc = history_A1_2d.history['accuracy']
        val_acc = history_A1_2d.history['val_accuracy']
        loss = history_A1_2d.history['loss']
        val_loss = history_A1_2d.history['val_loss']
        epochs_range = range(1, len(history_A1_2d.epoch) + 1)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Train Set')
        plt.plot(epochs_range, val_acc, label='Validation Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Train Set')
        plt.plot(epochs_range, val_loss, label='Validation Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.savefig(os.path.join(os.getcwd(), os.path.join('A1', 'A1_cnn.jpg')))
        plt.close()
        plt.figure()
        plt.plot(epochs_range, acc, label='Train Set')
        plt.plot(epochs_range, val_acc, label='Validation Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.savefig(os.path.join(os.getcwd(), os.path.join('A1', 'A1_cnn_acc.jpg')))
        plt.close()
