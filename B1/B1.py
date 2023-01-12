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
    #get directory
    if google:
        cur_dir = os.path.join('/content/drive/MyDrive/AMLS_assignment22_23/Datasets', name)
    else:
        cur_dir = os.path.join(os.getcwd(), 'Datasets/' + name)
    lsdir = os.listdir(cur_dir + '/img')
    last_name = '.' + lsdir[0].split('.')[1]
    #get image
    img_1d = []
    img = []
    #nums = min(len(lsdir), 10000)
    for i in range(len(lsdir)):
        img_1d.append(np.array(Image.open(cur_dir + '/img/' + str(i) + last_name)
                               .convert('L').resize((64, 64))).flatten())
        img.append(np.array(Image.open(cur_dir + '/img/' + str(i) + last_name)
                            .resize((64, 64))))
      #X.append(cv2.imread(cur_dir + '/img/' + str(i) + last_name, cv2.IMREAD_GRAYSCALE).flatten())
    #get labels of two tasks
    labels = pd.read_csv(os.path.join(cur_dir, 'labels.csv'))
    labels_first = []
    #labels_second = []
    for i in range(len(labels)):
        label = labels[labels.columns[0]][i].split('\t')
        labels_first.append(label[2])
        #labels_second.append(label[1])
    return np.array(img_1d), np.array(labels_first), img #, np.array(labels_second)

def count_labels(label):
    labels_b1 = []
    for i in label:
        if i not in labels_b1:
            labels_b1.append(i)
    return len(labels_b1)

def KNNClassifier(x_train, y_train, x_test, k):
    # Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)  # Fit KNN model
    y_pred = neigh.predict(x_test)
    return y_pred

def img_SVM(training_images, training_labels, test_images, test_labels, titles, C, task):
    #classifier = ...
    models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(training_images, training_labels) for clf in models)
    pred = []
    for model, i, title in zip(models, range(2), titles):
        #classifier = model
        #classifier.fit(training_images, training_labels)
        pred.append(model.predict(test_images))
        print(task + ': ' + title + ' ' + "Accuracy:", accuracy_score(test_labels, pred[i]))

def get_B1_results():
    google_drive = False
    datasets = ['celeba', 'celeba_test', 'cartoon_set', 'cartoon_set_test']
    X_car, y_car_B1, X_car_fig = get_data_label_taskb(datasets[2], google_drive)
    X_car_test, y_car_test_B1, X_car_test_fig = get_data_label_taskb(datasets[3], google_drive)
    print('The number of Task B1 labels is ' + str(count_labels(y_car_B1)))
    '''
    #Task B1
    knn_accuracy = list()
    max_knn = [0, 0]#max_knn = [value, index]
    for i in range(1, 41):
        y_pred_B1 = KNNClassifier(X_car, y_car_B1, X_car_test, i)
        knn_accuracy.append(float(np.sum(y_pred_B1 == y_car_test_B1)) / len(y_pred_B1))
        if max_knn[0] < knn_accuracy[-1]:
            max_knn[0] = knn_accuracy[-1]
            max_knn[1] = i

    if os.path.exists(os.path.join(os.getcwd(), 'B1')):
        plt.figure()
        plt.plot(range(1,41), knn_accuracy, label='accuracy')
        plt.scatter(max_knn[1], max_knn[0], color='r', label='best point')
        plt.title('The accuracy of KNNs')
        plt.xlabel('value of k')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), os.path.join('B1', 'B1_knn.jpg')))
        plt.close()

    y_pred_B1 = KNNClassifier(X_car, y_car_B1, X_car_test, max_knn[1])
    score_B1 = metrics.accuracy_score(y_car_test_B1, y_pred_B1)
    print(score_B1)
    num_correct = np.sum(y_pred_B1 == y_car_test_B1)
    accuracy = float(num_correct) / len(y_pred_B1)
    print('Task B1 Got %d / %d correct => accuracy: %f (k = %d)' % (num_correct, len(y_pred_B1), accuracy, max_knn[1]))

    #Task B1
    rf_accuracy = list()
    max_rf = [0, 0]#max_rf = [value, index]
    for i in range(10,130,10):
        clf_temp = RandomForestClassifier(n_estimators=i)
        clf_temp.fit(X_car, y_car_B1)
        rf_accuracy.append(accuracy_score(y_car_test_B1, clf_temp.predict(X_car_test)))
        if max_rf[0] < rf_accuracy[-1]:
            max_rf[0] = rf_accuracy[-1]
            max_rf[1] = i
    if os.path.exists(os.path.join(os.getcwd(), 'B1')):
        plt.figure()
        plt.plot(range(10,130,10), rf_accuracy, label='accuracy')
        plt.scatter(max_rf[1], max_rf[0], color='r', label='best point')
        plt.title('The accuracy of Random Forests')
        plt.xlabel('number of estimators')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), os.path.join('B1', 'B1_rf.jpg')))
        plt.close()
    # create the model - Random Forests
    clf_B1  = RandomForestClassifier(n_estimators=max_rf[1])
    # fit the training data to the model
    clf_B1.fit(X_car, y_car_B1)
    #print(clf.fit(trainDataGlobal, trainLabelsGlobal))
    clf_pred_B1 = clf_B1.predict(X_car_test)
    #clf_pred = clf.predict(global_feature.reshape(1,-1))[0]
    print("Task B1 Random Forest test Accuracy: %f (number of weak learners = %d)"% (accuracy_score(y_car_test_B1, clf_pred_B1), max_rf[1]))

    # sklearn functions implementation
    titles = ('SVC with linear kernel',
            'SVC with polynomial (degree 3) kernel')
    pred_car_B1 = img_SVM(X_car, y_car_B1, X_car_test, y_car_test_B1, titles, C=1, task = 'Task_B1')
    '''

    X_train_B_fig = tf.stack([(i).astype("float64") for i in X_car_fig])
    X_val_B_fig = tf.stack([(i).astype("float64") for i in X_car_test_fig[:1250]])
    X_test_B_fig = tf.stack([(i).astype("float64") for i in X_car_test_fig[1250:]])

    y_train_B1 = [float(i) for i in y_car_B1]
    y_val_B1 = [float(i) for i in y_car_test_B1[:1250]]
    y_test_B1 = [float(i) for i in y_car_test_B1[1250:]]
    y_train_B1 = tf.stack(y_train_B1)
    y_val_B1 = tf.stack(y_val_B1)
    y_test_B1 = tf.stack(y_test_B1)


    model_B1 = models.Sequential()
    model_B1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 4)))
    model_B1.add(layers.MaxPooling2D((2, 2)))
    model_B1.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_B1.add(layers.MaxPooling2D((2, 2)))
    model_B1.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_B1.add(layers.Flatten())
    model_B1.add(layers.Dense(64, activation='relu'))
    model_B1.add(layers.Dense(10))
    model_B1.summary()
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.B1.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)
    ]
    BATCH_SIZE = 500
    EPOCHS = 25
    model_B1.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    history_B1_2d = model_B1.fit(X_train_B_fig, y_train_B1,
                                epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_data=(X_val_B_fig, y_val_B1),
                                callbacks=callbacks_list)
    model_best = load_model('best_model.B1.h5')
    print('The accuracy of Task B1 by CNN is %.3f' % model_best.evaluate(X_test_B_fig, y_test_B1, verbose=0)[1])

    
    if os.path.exists(os.path.join(os.getcwd(), 'B1')):
        acc = history_B1_2d.history['accuracy']
        val_acc = history_B1_2d.history['val_accuracy']
        loss = history_B1_2d.history['loss']
        val_loss = history_B1_2d.history['val_loss']
        epochs_range = range(1, len(history_B1_2d.epoch) + 1)
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
        plt.savefig(os.path.join(os.getcwd(), os.path.join('B1', 'B1_cnn.jpg')))
        plt.close()
        plt.figure()
        plt.plot(epochs_range, acc, label='Train Set')
        plt.plot(epochs_range, val_acc, label='Validation Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.savefig(os.path.join(os.getcwd(), os.path.join('B1', 'B1_cnn_acc.jpg')))
        plt.close()
