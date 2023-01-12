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
    for i in range(len(lsdir)):
        img_1d.append(np.array(Image.open(cur_dir + '/img/' + str(i) + last_name)
                               .convert('L').resize((64, 64))).flatten())
        img.append(np.array(Image.open(cur_dir + '/img/' + str(i) + last_name)
                            .convert('L').resize((64, 64))))
      #X.append(cv2.imread(cur_dir + '/img/' + str(i) + last_name, cv2.IMREAD_GRAYSCALE).flatten())
    #get labels of two tasks
    labels = pd.read_csv(os.path.join(cur_dir, 'labels.csv'))
    labels_second = []
    for i in range(len(labels)):
        label = labels[labels.columns[0]][i].split('\t')
        labels_second.append(label[3])
    return np.array(img_1d), np.array(labels_second), img

def KNNClassifier(X_train, y_train, X_test, k):
    # Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)  # Fit KNN model
    Y_pred = neigh.predict(X_test)
    return Y_pred

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
   # print(pred)
    return pred
def get_A2_results():
    datasets = ['celeba', 'celeba_test', 'cartoon_set', 'cartoon_set_test']
    ###X_cel.shape = (218,178,3), X_car.shape = (500,500,4)
    google_drive = False
    X_cel, y_cel_A2, X_cel_fig = get_data_label(datasets[0], google_drive)
    X_cel_test, y_cel_test_A2, X_cel_test_fig = get_data_label(datasets[1], google_drive)
    '''
    #Task A2
    knn_accuracy = list()
    max_knn = [0, 0]#max_knn = [value, index]
    for i in range(1, 16):
        y_pred_A2 = KNNClassifier(X_cel, y_cel_A2, X_cel_test, i)
        knn_accuracy.append(float(np.sum(y_pred_A2 == y_cel_test_A2)) / len(y_pred_A2))
        if max_knn[0] < knn_accuracy[-1]:
            max_knn[0] = knn_accuracy[-1]
            max_knn[1] = i

    if os.path.exists(os.path.join(os.getcwd(), 'A2')):
        plt.figure()
        plt.plot(range(1,16), knn_accuracy, label='accuracy')
        plt.scatter(max_knn[1], max_knn[0], color='r', label='best point')
        plt.title('The accuracy of KNNs')
        plt.xlabel('value of k')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), os.path.join('A2', 'A2_knn.jpg')))
        plt.close()

    y_pred_A2 = KNNClassifier(X_cel, y_cel_A2, X_cel_test, max_knn[1])
    score_A2 = metrics.accuracy_score(y_cel_test_A2, y_pred_A2)
    print(score_A2)
    num_correct = np.sum(y_pred_A2 == y_cel_test_A2)
    accuracy = float(num_correct) / len(y_pred_A2)
    print('Task A2 Got %d / %d correct => accuracy: %f(k = %d)' % (num_correct, len(y_pred_A2), accuracy, max_knn[1]))



    #Task A2
    rf_accuracy = list()
    max_rf = [0, 0]#max_rf = [value, index]
    for i in range(10,130,10):
        clf_temp = RandomForestClassifier(n_estimators=i)
        clf_temp.fit(X_cel, y_cel_A2)
        rf_accuracy.append(accuracy_score(y_cel_test_A2, clf_temp.predict(X_cel_test)))
        if max_rf[0] < rf_accuracy[-1]:
            max_rf[0] = rf_accuracy[-1]
            max_rf[1] = i
    if os.path.exists(os.path.join(os.getcwd(), 'A2')):
        plt.figure()
        plt.plot(range(10,130,10), rf_accuracy, label='accuracy')
        plt.scatter(max_rf[1], max_rf[0], color='r', label='best point')
        plt.title('The accuracy of Random Forests')
        plt.xlabel('number of estimators')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), os.path.join('A2', 'A2_rf.jpg')))
        plt.close()

    # create the model - Random Forests
    clf_A2  = RandomForestClassifier(n_estimators=max_rf[1])
    # fit the training data to the model
    clf_A2.fit(X_cel, y_cel_A2)
    #print(clf.fit(trainDataGlobal, trainLabelsGlobal))
    clf_pred_A2 = clf_A2.predict(X_cel_test)
    #clf_pred = clf.predict(global_feature.reshape(1,-1))[0]
    print("Task A2 Random Forest test Accuracy: %f (number of weak learners = %d)"% (accuracy_score(y_cel_test_A2, clf_pred_A2), max_rf[1]))

    # sklearn functions implementation
    titles = ('SVC with linear kernel',
            'SVC with polynomial (degree 3) kernel')
    pred_cel_A2 = img_SVM(X_cel, y_cel_A2, X_cel_test, y_cel_test_A2, titles, C=1, task = 'Task_A2')
    '''
    #Task A2
    X_train_A_fig = tf.stack([(i/255).astype("float64") for i in X_cel_fig[:4500]])
    X_val_A_fig = tf.stack([(i/255).astype("float64") for i in X_cel_fig[4500:5000]])
    X_test_A_fig = tf.stack([(i/255).astype("float64") for i in X_cel_test_fig])

    y_train_A2 = [i == '1' for i in y_cel_A2[:4500]]
    y_val_A2 = [i == '1' for i in y_cel_A2[4500:5000]]
    y_test_A2 = [i == '1' for i in y_cel_test_A2]
    y_train_A2 = tf.stack(y_train_A2)
    y_val_A2 = tf.stack(y_val_A2)
    y_test_A2 = tf.stack(y_test_A2)

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

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.A2.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)
    ]
    BATCH_SIZE = 500
    EPOCHS = 25
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history_A2_2d = model.fit(X_train_A_fig, y_train_A2,
                            epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(X_val_A_fig, y_val_A2),
                            callbacks=callbacks_list)
    model_best = load_model('best_model.A2.h5')
    print('The accuracy of Task A2 by CNN is %.3f' % model_best.evaluate(X_test_A_fig, y_test_A2, verbose=0)[1])

    if os.path.exists(os.path.join(os.getcwd(), 'A2')):
        acc = history_A2_2d.history['accuracy']
        val_acc = history_A2_2d.history['val_accuracy']
        loss = history_A2_2d.history['loss']
        val_loss = history_A2_2d.history['val_loss']
        epochs_range = range(1, len(history_A2_2d.epoch) + 1)
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
        plt.savefig(os.path.join(os.getcwd(), os.path.join('A2', 'A2_cnn.jpg')))
        plt.close()
        plt.figure()
        plt.plot(epochs_range, acc, label='Train Set')
        plt.plot(epochs_range, val_acc, label='Validation Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.savefig(os.path.join(os.getcwd(), os.path.join('A2', 'A2_cnn_acc.jpg')))
        plt.close()
