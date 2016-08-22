"""
Train a classifier using the Bag of Features approach
Extract SIFT features for each training image and form a vocabulary
Author: Chandni Shankar
"""
import cv2
import numpy as np
import os, sys
from os.path import join
from cv2 import BOWKMeansTrainer
from cv2 import BOWImgDescriptorExtractor
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import CreateInstance
from sklearn import neighbors
from sklearn import ensemble
from sklearn import cluster

def findSift(trainDir, printKeypointImages=False):
    clusterCount = 600
    allInstances = CreateInstance.getInstances(trainDir)
    print "Sift done"
    bow = BOWKMeansTrainer(clusterCount)
    for instance in allInstances:
        bow.add(instance.descriptors)

        if (printKeypointImages == "keypoints"):
            kpDir = join(os.path.dirname(trainDir), 'KeyPoints')
            if not (os.path.exists(kpDir)):
                os.makedirs(kpDir)
            objectName = instance.actualLabel + "_" + str(instance.id) + ".jpg"
            imagePath = join(kpDir, objectName)
            if not (os.path.exists(imagePath)):
                cv2.imwrite(imagePath, instance.keyPointImage)

    #KMEans clustering
    vocabulary = bow.cluster()

    #Histogram of features
    siftExtractor = cv2.DescriptorExtractor_create("SIFT")
    # Initiailizing parameters for creating a Flann matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)

    BOWextractor = BOWImgDescriptorExtractor(siftExtractor, flannMatcher)
    BOWextractor.setVocabulary(vocabulary)
    detector = cv2.FeatureDetector_create("SIFT")
    trainData = []
    labelsList = []
    for instance in allInstances:
        kp = detector.detect(instance.grayOrigImg)
        des = BOWextractor.compute(instance.grayOrigImg, kp)
        trainData.append(des)
        labelsList.append(instance.actualLabel)

    #Visual Vocabulary
    features = np.array(trainData)
    labels = np.array(labelsList)

    #Converting a 3D descriptor to 2D
    nsamples, nx, ny = features.shape
    features_2d = features.reshape((nsamples, nx * ny))

    # Scaling the words
    stdSlr = StandardScaler().fit(features_2d)
    features = stdSlr.transform(features_2d)

    joblib.dump((features, labels), "trainFeatures.pkl", compress=3)

    print "features ready"

    # Train the Linear SVM
    clf = LinearSVC(C=5)
    clf.fit(features, labels)
    classifier = 'linearSVM'
    vocabName = classifier +"_bof.pkl"
    # Save the SVM
    joblib.dump((clf, stdSlr, clusterCount, vocabulary), vocabName, compress=3)

    print "Linear SVM Training Completed!"



    clf = svm.SVC(kernel='rbf')
    clf.fit(features, labels)
    classifier = 'rbfkernel'
    vocabName = classifier + "_bof.pkl"
    # Save the SVM
    joblib.dump((clf, stdSlr, clusterCount, vocabulary), vocabName, compress=3)

    print "RBF Training Completed!!"


    clf = neighbors.KNeighborsClassifier(10, weights='uniform')
    clf.fit(features, labels)
    classifier = 'knn'
    vocabName = classifier + "_bof.pkl"
    # Save the SVM
    joblib.dump((clf, stdSlr, clusterCount, vocabulary), vocabName, compress=3)

    print "KNN Training Completed!!"


    clf = ensemble.RandomForestClassifier(n_estimators = 100)
    clf.fit(features, labels)
    classifier = 'RandomForest'
    vocabName = classifier + "_bof.pkl"
    # Save the SVM
    joblib.dump((clf, stdSlr, clusterCount, vocabulary), vocabName, compress=3)

    print "Random Forest Training Completed!!"



def main(argv):
    trainDir = argv[0]
    if len(argv) > 1:
        keyPointImages = argv[1]
        findSift(trainDir, keyPointImages)
    else:
        findSift(trainDir)


if __name__ == "__main__":
    main(sys.argv[1:])
