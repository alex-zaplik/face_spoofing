import cv2
import os
import random

from facenorm import FaceNormalizer


# Loading data
def loadFromList(listFilePath, outPath, outName, histFunc, grayscale, fn, label):
    hists = []

    os.makedirs(outPath, exist_ok=True)
    output = open(os.path.join(outPath, outName + ".txt"), "w")

    clientTrainList = open(listFilePath, "r")
    for path in clientTrainList:
        fullPath = os.path.join(os.path.split(listFilePath)[0], path.strip())
        img = cv2.imread(fullPath)
        faces = fn.normalizedFaces(img, grayscale=grayscale)

        for face in faces:
            hist = histFunc(face)
            output.write(str(hist) + '\n')
            hists.append([hist, label])

        cv2.waitKey(0)
    return hists


# Load default training data
def getTrainingData(dataPath, dataPrefix, dataSuffix, listPathTrue, listPathSpoof, histFunc, grayscale):
    fn = FaceNormalizer()

    print("\tLoading client data...")
    data = loadFromList(listPathTrue, dataPath, dataPrefix + "_client_" + dataSuffix, histFunc, grayscale, fn, 0)

    print("\tLoading imposter data...")
    data.extend(loadFromList(listPathSpoof, dataPath, dataPrefix + "_imposter_" + dataSuffix, histFunc, grayscale, fn, 1))

    random.shuffle(data)
    return [d[0] for d in data], [d[1] for d in data]


def loadDataFile(filePath, label):
    hists = []
    file = open(filePath, "r")
    
    for hist in file:
        hists.append((eval(hist), label))

    return hists


def getTrainingDataFromFile(path, prefix, suffix):
    print("\tLoading client data...")
    data = loadDataFile(os.path.join(path, prefix + "_client_" + suffix + ".txt"), 0)

    print("\tLoading imposter data...")
    data.extend(loadDataFile(os.path.join(path, prefix + "_imposter_" + suffix + ".txt"), 1))

    random.shuffle(data)
    return [d[0] for d in data], [d[1] for d in data]

