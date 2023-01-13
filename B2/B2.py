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

def get_data_label_taskb(name, google):
    # This function is to get data and corresponding labels
    # get datasets directory
    if google:
        cur_dir = os.path.join('/content/drive/MyDrive/AMLS_assignment22_23/Datasets', name)
    else:
        cur_dir = os.path.join(os.getcwd(), 'Datasets/' + name)
    # get file name and its last name (.jpg or .png)
    lsdir = os.listdir(cur_dir + '/img')
    last_name = '.' + lsdir[0].split('.')[1]
    # get image, using img_1d and img to store 1D image sequences and 3D color image
    img_1d = []
    img = []
    for i in range(len(lsdir)):
        img_1d.append(np.array(Image.open(cur_dir + '/img/' + str(i) + last_name)
                               .convert('L').resize((64, 64))).flatten())
        img.append(np.array(Image.open(cur_dir + '/img/' + str(i) + last_name)
                            .resize((64, 64))))
    # load data label file and seperate labels of Task B1 from it
    labels = pd.read_csv(os.path.join(cur_dir, 'labels.csv'))
    labels_second = []
    for i in range(len(labels)):
        label = labels[labels.columns[0]][i].split('\t')
        labels_second.append(label[1])
    return np.array(img_1d), np.array(labels_second), img # np.array(labels_first)

def count_labels(label):
    # get the number of category of the data
    labels_b1 = []
    for i in label:
        if i not in labels_b1:
            labels_b1.append(i)
    return len(labels_b1)

def KNNClassifier(x_train, y_train, x_test, k):
    # This function is to construct KNN model and get the prediction results
    # Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    # Fit KNN model
    neigh.fit(x_train, y_train)
    # get the prediction result of KNN under the value of k
    y_pred = neigh.predict(x_test)
    return y_pred

def img_SVM(training_images, training_labels, test_images, test_labels, titles, C):
    # This function is to construct SVMs model and get the prediction results
    # define model kernels and pack them together after fitting
    models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(training_images, training_labels) for clf in models)
    pred = []
    # get the results of svm models that use different kernels
    for model, i, title in zip(models, range(2), titles):
        pred.append(model.predict(test_images))
        print(title + ' ' + "Accuracy:", accuracy_score(test_labels, pred[i]))
    return pred

def get_B2_results():
    # This function is the main function to solve Task B2
    # whether use google_drive, if use, google_drive = True, if not, google_drive = False
    google_drive = False
    # Get dataset names
    datasets = ['celeba', 'celeba_test', 'cartoon_set', 'cartoon_set_test']
    # X_car is 1D image sequences, y_car_B1 is the labels of Task B2, X_car_fig is 3D color images
    X_car, y_car_B2, X_car_fig = get_data_label_taskb(datasets[2], google_drive)
    # X_car_test, y_car_test_B1, X_car_test_fig are corresponding test set
    X_car_test, y_car_test_B2, X_car_test_fig = get_data_label_taskb(datasets[3], google_drive)
    # print the number of category
    print('The number of Task B2 labels is ' + str(count_labels(y_car_B2)))
    
    # Task B2 KNN
    # document knn accuracy
    knn_accuracy = list()
    # document the value of k under the maximum of accuracy
    max_knn = [0, 0]#max_knn = [value, index]
    # find the maximum accuracy of KNN from 1 to 40 of the value of k
    for i in range(1, 41):
        y_pred_B2 = KNNClassifier(X_car, y_car_B2, X_car_test, i)
        knn_accuracy.append(float(np.sum(y_pred_B2 == y_car_test_B2)) / len(y_pred_B2))
        if max_knn[0] < knn_accuracy[-1]:
            max_knn[0] = knn_accuracy[-1]
            max_knn[1] = i
    # Visualise the results of KNN
    if os.path.exists(os.path.join(os.getcwd(), 'B2')):
        plt.figure()
        plt.plot(range(1,41), knn_accuracy, label='accuracy')
        plt.scatter(max_knn[1], max_knn[0], color='r', label='best point')
        plt.title('The accuracy of KNNs')
        plt.xlabel('value of k')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), os.path.join('B2', 'B2_knn.jpg')))
        plt.close()
    # Using the best value of k to output its accuracy
    y_pred_B2 = KNNClassifier(X_car, y_car_B2, X_car_test, max_knn[1])
    score_B2 = metrics.accuracy_score(y_car_test_B2, y_pred_B2)
    print(score_B2)
    num_correct = np.sum(y_pred_B2 == y_car_test_B2)
    accuracy = float(num_correct) / len(y_pred_B2)
    print('Task B2 Got %d / %d correct => accuracy: %f (k = %d)' % (num_correct, len(y_pred_B2), accuracy, max_knn[1]))

    #Task B2 Random Forest
    # document rf accuracy
    rf_accuracy = list()
    # document the number of weak learners under the maximum accuracy of rf model
    max_rf = [0, 0]# max_rf = [value, index]
    # find the maximum accuracy of rf from 10 to 120 of the number of weak learners, the step is 10
    for i in range(10,130,10):
        clf_temp = RandomForestClassifier(n_estimators=i)
        clf_temp.fit(X_car, y_car_B2)
        rf_accuracy.append(accuracy_score(y_car_test_B2, clf_temp.predict(X_car_test)))
        if max_rf[0] < rf_accuracy[-1]:
            max_rf[0] = rf_accuracy[-1]
            max_rf[1] = i
    # Visualise the results of rf
    if os.path.exists(os.path.join(os.getcwd(), 'B2')):
        plt.figure()
        plt.plot(range(10,130,10), rf_accuracy, label='accuracy')
        plt.scatter(max_rf[1], max_rf[0], color='r', label='best point')
        plt.title('The accuracy of Random Forests')
        plt.xlabel('number of estimators')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), os.path.join('B2', 'B2_rf.jpg')))
        plt.close()

    # create the model - Random Forests
    clf_B2  = RandomForestClassifier(n_estimators=max_rf[1])
    # fit the training data to the model
    clf_B2.fit(X_car, y_car_B2)
    # get the prediction of the results under the best number of weak learners
    # (retraining the rf, the best value may change)
    clf_pred_B2 = clf_B2.predict(X_car_test)
    print("Task B2 Random Forest test Accuracy: %f (number of weak learners = %d)"% (accuracy_score(y_car_test_B2, clf_pred_B2), max_rf[1]))

    # Task B1 CNN
    # Split dataset: train, val, test
    X_train_B_fig = tf.stack([(i).astype("float64") for i in X_car_fig])
    X_val_B_fig = tf.stack([(i).astype("float64") for i in X_car_test_fig[:1250]])
    X_test_B_fig = tf.stack([(i).astype("float64") for i in X_car_test_fig[1250:]])
    # split labels and preprocess them
    y_train_B2 = [float(i) for i in y_car_B2]
    y_val_B2 = [float(i) for i in y_car_test_B2[:1250]]
    y_test_B2 = [float(i) for i in y_car_test_B2[1250:]]
    y_train_B2 = tf.stack(y_train_B2)
    y_val_B2 = tf.stack(y_val_B2)
    y_test_B2 = tf.stack(y_test_B2)
    # construct cnn model
    model_B2 = models.Sequential()
    model_B2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 4)))
    model_B2.add(layers.MaxPooling2D((2, 2)))
    model_B2.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_B2.add(layers.MaxPooling2D((2, 2)))
    model_B2.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_B2.add(layers.Flatten())
    model_B2.add(layers.Dense(64, activation='relu'))
    model_B2.add(layers.Dense(10))
    model_B2.summary()
    # add callbacks to optimise the training process
    # monitor loss to save the best model
    # set earlystopping monitoring accuracy and its patience is 2
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.B2.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)
    ]
    BATCH_SIZE = 500
    EPOCHS = 25
    # configure the model for training
    model_B2.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    # fit the model and save the best model in the same time
    history_B2_2d = model_B2.fit(X_train_B_fig, y_train_B2,
                                epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_data=(X_val_B_fig, y_val_B2),
                                callbacks=callbacks_list)
    # load the best model to do classification and output the results
    model_best = load_model('best_model.B2.h5')
    print('The accuracy of Task B2 by CNN is %.3f' % model_best.evaluate(X_test_B_fig, y_test_B2, verbose=0)[1])
    # visualise the training process and save it to local
    if os.path.exists(os.path.join(os.getcwd(), 'B2')):
        acc = history_B2_2d.history['accuracy']
        val_acc = history_B2_2d.history['val_accuracy']
        loss = history_B2_2d.history['loss']
        val_loss = history_B2_2d.history['val_loss']
        epochs_range = range(1, len(history_B2_2d.epoch) + 1)
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
        plt.savefig(os.path.join(os.getcwd(), os.path.join('B2', 'B2_cnn.jpg')))
        plt.close()
        plt.figure()
        plt.plot(epochs_range, acc, label='Train Set')
        plt.plot(epochs_range, val_acc, label='Validation Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.savefig(os.path.join(os.getcwd(), os.path.join('B2', 'B2_cnn_acc.jpg')))
        plt.close()