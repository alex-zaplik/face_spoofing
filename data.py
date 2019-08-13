import cv2
import os
import random


# Loading data
def loadFromList(basePath, listFilePath, imageFolderPath, label, outPrefix, histFunc, grayscale):
    hists = []
    output = open(outPrefix + "_" + listFilePath, "w")
    clientTrainList = open(os.path.join(basePath, listFilePath), "r")
    for path in clientTrainList:
        fullPath = os.path.join(basePath, imageFolderPath, path.strip())
        
        img = false
        if grayscale:
            img = cv2.imread(fullPath, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(fullPath, cv2.IMREAD_COLOR)
        
        hist = histFunc(img)
        output.write(str(hist) + '\n')
        hists.append([hist, label])
    return hists


# Load default training data
def getTrainingData(basePath, infix, outPrefix, histFunc):
    print("\tLoading client data...")
    data = loadFromList(basePath, "client_" + infix + "_normalized.txt", "ClientNormalized", 0, outPrefix, histFunc)

    print("\tLoading imposter data...")
    data.extend(loadFromList(basePath, "imposter_" + infix + "_normalized.txt", "ImposterNormalized", 1, outPrefix, histFunc))

    random.shuffle(data)
    return [d[0] for d in data], [d[1] for d in data]


def loadDataFile(filePath, label):
    hists = []
    file = open(filePath, "r")
    
    for hist in file:
        hists.append((eval(hist), label))

    return hists


def getTrainingDataFromFile(prefix, infix):
    print("\tLoading client data...")
    data = loadDataFile(prefix + "_client_" + infix + "_normalized.txt", 0)

    print("\tLoading imposter data...")
    data.extend(loadDataFile(prefix + "_imposter_" + infix + "_normalized.txt", 1))

    random.shuffle(data)
    return [d[0] for d in data], [d[1] for d in data]

