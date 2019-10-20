import cv2
import os
import random
import joblib

from facenorm import FaceNormalizer


def loadPhoto(path, histFunc, grayscale):
    """This function loads a photo into memory, detects faces in it and calculates
    histograms for each of them used the method given

    Args:
        path (str): The path to the image
        histFunc (TODO): The method of calculating histograms. Must be a callable
        taking the image as the only argument and returning a histogram
    
    Returns:
        list(list(int)): The calculated histograms
    """

    fn = FaceNormalizer()
    hists = []

    img = cv2.imread(path)
    faces = fn.normalizedFaces(img, grayscale=grayscale, returnBox=True)

    for face, box in faces:
        hist = histFunc(face)
        hists.append([hist, box])
    
    return hists

# Loading data
def loadFromList(listFilePath, outPath, outName, histFunc, grayscale, fn, label, toFile=True):
    """
    TODO
    """
    
    hists = []

    if toFile:
        os.makedirs(outPath, exist_ok=True)
        output = open(os.path.join(outPath, outName + ".txt"), "w")

    clientTrainList = open(listFilePath, "r")
    for path in clientTrainList:
        fullPath = os.path.join(os.path.split(listFilePath)[0], path.strip())
        img = cv2.imread(fullPath)
        faces = fn.normalizedFaces(img, grayscale=grayscale)

        for face in faces:
            hist = histFunc(face)
            if toFile:
                output.write(str(hist) + '\n')
            hists.append([hist, label])
    
    return hists


# Load default training data
def getTrainingData(dataPath, dataPrefix, dataSuffix, listPathTrue, listPathSpoof, histFunc, grayscale):
    """
    TODO
    """

    fn = FaceNormalizer()

    print("\tLoading client data...")
    data = loadFromList(listPathTrue, dataPath, dataPrefix + "_client_" + dataSuffix, histFunc, grayscale, fn, 0)

    print("\tLoading imposter data...")
    data.extend(loadFromList(listPathSpoof, dataPath, dataPrefix + "_imposter_" + dataSuffix, histFunc, grayscale, fn, 1))

    random.shuffle(data)
    return [d[0] for d in data], [d[1] for d in data]


def loadDataFile(filePath, label):
    """
    TODO
    """

    hists = []
    file = open(filePath, "r")
    
    for hist in file:
        hists.append((eval(hist), label))

    return hists


def getTrainingDataFromFile(path, prefix, suffix):
    """
    TODO
    """

    print("\tLoading client data...")
    data = loadDataFile(os.path.join(path, prefix + "_client_" + suffix + ".txt"), 0)

    print("\tLoading imposter data...")
    data.extend(loadDataFile(os.path.join(path, prefix + "_imposter_" + suffix + ".txt"), 1))

    random.shuffle(data)
    return [d[0] for d in data], [d[1] for d in data]


def saveClassifier(clf, threshold, method, path):
    """Saves an SVM classifier to a binary file at 'path' along with its EER threshold
    and the method that was used to train it

    Args:
        clf (SVM): The classifier
        threshold (float): The EER threshold
        method (str): The method that was used
        path (str): The path where the SVM will be saved
    """

    joblib.dump((clf, threshold, method), path)


def loadClassifier(path):
    """Loads an SVM classifier from a binary file at 'path' along with its EER threshold
    and the method that was used to train it

    Args:
        path (str): The path to the SVM file

    Retuns:
        (SVM, float, str): The classifier, the EER threshold and the method that was used
    """

    (clf, threshold, method) = joblib.load(path)
    return clf, threshold, method