"""
Predict the label of the images using the vocabulary of the trained classifier
Author: Chandni Shankar
"""
import cv2
import numpy as np
import os, sys
from os.path import join
from cv2 import BOWKMeansTrainer
from cv2 import BOWImgDescriptorExtractor
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import CreateInstance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def getAccuracy(labelsList, predictions):
    index, accuracy = 0, 0
    for prediction in predictions:
        if (prediction == labelsList[index]):
            accuracy += 1
        index += 1
    accuracy = float(accuracy) / float(len(labelsList)) * 100
    return accuracy


def labelImages(allInstances, predictions, classifier):
    # Visualize the results, if "visualize" flag set to true by the user
    index = 0
    for prediction in predictions:
        imagepath = allInstances[index].imagePath
        image = cv2.imread(imagepath)
        newDirName = join(os.path.dirname(os.path.dirname(os.path.dirname(imagepath))),
                          ('LabelledImages_' + classifier))
        if not os.path.exists(newDirName):
            os.makedirs(newDirName)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
        cv2.imwrite(join(newDirName, imagepath.split("\\")[-1]), image)
        index += 1

def plotCM(predictions,labelsList, classes, classifier):
    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Spectral):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, list(classes), rotation=45)
        plt.yticks(tick_marks, list(classes))
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix
    cm = confusion_matrix(labelsList, predictions)
    np.set_printoptions(precision=2)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    name = classifier + "_cm.jpg"
    plt.savefig(name)
    plt.show()

def predict(testDir, visualize = False, plot = False):
    classifier = "linearSVM"
    vocabName = classifier+'_bof.pkl'
    #retrieve the svm data

    #Instances of Test directory
    allInstances = CreateInstance.getInstances(testDir)

    clf, stdSlr, k, voc = joblib.load(vocabName)
    # Extract feature keypoints for test images and perform matching to find distinctive descriptors
    siftExtractor = cv2.DescriptorExtractor_create("SIFT")
    # Initiailizing parameters for creating a Flann matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)

    BOWextractor = BOWImgDescriptorExtractor(siftExtractor, flannMatcher)
    BOWextractor.setVocabulary(voc)
    testData = []
    labelsList = []
    classes = set()
    for instance in allInstances:
        des = BOWextractor.compute(instance.grayOrigImg, instance.imageKeypoints)
        testData.append(des)
        labelsList.append(instance.actualLabel)
        classes.add(instance.actualLabel)
    features = np.array(testData)
    labels = np.array(labelsList)

    nsamples, nx, ny = features.shape
    test_features = features.reshape((nsamples, nx * ny))

    # Scale the features
    test_features = stdSlr.transform(test_features)

    joblib.dump((test_features,labels), "testFeatures.pkl", compress=3)

    # Perform the predictions
    predictions = clf.predict(test_features)


    classifiers = ['linearSVM','rbfkernel', 'knn','RandomForest']
    accuracy = getAccuracy(labelsList, predictions)
    print ("Accuracy " + classifier), accuracy

    if (visualize == "visualize"):
        labelImages(allInstances, predictions, classifier)
    if (plot == "plot"):
        plotCM(predictions, labelsList, classes, classifier)

    for classifier in classifiers[1:]:
        vocabName = classifier+'_bof.pkl'
        clf, stdSlr, k, voc = joblib.load(vocabName)
        predictions = clf.predict(test_features)
        accuracy = getAccuracy(labelsList, predictions)
        print ("Accuracy "+classifier), accuracy
        if (visualize == "visualize"):
            labelImages(allInstances, predictions, classifier)
        if (plot == "plot"):
            plotCM(predictions, labelsList, classes, classifier)




def main(argv):
    testDir = argv[0]
    if len(argv) > 1:
        visualize = argv[1]
        plot = argv[2]
        predict(testDir, visualize, plot)
    else:
        predict(testDir)


if __name__ == "__main__":
    main(sys.argv[1:])
