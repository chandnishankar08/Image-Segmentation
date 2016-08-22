"""
Class to make a list of instances with SIFT feature information for the directory
Author: Chandni Shankar
"""
import cv2
import os, sys
from os.path import join

class Instances:
    """
    Class to store information about each image in the given directory
    """
    def __init__(self, label, keyPoints, descriptors, origImg, kpImg, idNo, imagePath):
        self.actualLabel = label
        self.imageKeypoints = keyPoints
        self.descriptors = descriptors
        self.id = idNo
        self.grayOrigImg = origImg
        self.keyPointImage = kpImg
        self.imagePath = imagePath

def getInstances(dir):
    """
    :param dir:training directory / test directory path
    :return: list of all instances with SIFT keypoints and descriptors
    """
    allInstances = []
    classes = []
    index, idNum = 0, 1
    for root, dirs, files in os.walk(dir):
        if dirs:
            classes = dirs
        elif files:
            label = classes[index].split(".")[-1]
            for file in files:
                img = cv2.imread(join(root, file))
                grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sift = cv2.SIFT()
                kp = sift.detect(grayImage, None)
                kpImg = cv2.drawKeypoints(grayImage, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                kp, des = sift.compute(grayImage, kp)
                allInstances.append(Instances(label, kp, des, grayImage, kpImg, idNum, join(root, file)))
                idNum += 1
            index += 1
    return allInstances
