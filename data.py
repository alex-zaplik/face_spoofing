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
        histFunc (callable): The method of calculating histograms. Must be a callable taking the image as the only argument and returning a histogram
    
    Returns:
        list(list((int, (int, int, int, int)))): The calculated histograms along with the detected face box
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
    """Load images from a single file containg relative paths, detect faces it those images,
    calculates feature histograms and saves the results to another file (this can be disabled).

    Args:
        listFilePath (str): Path the a file containg a list (one element per line) of relative paths to images that will be processed
        outPath (str): The path to a directory where the output file will be created
        outName (str): The name of the output file (.txt will be added to it)
        histFunc (callable): A function used to calculate a histogram for a single detected face
        grayscale (bool): If true the images will be converted to grayscale before processing (that is required by some histogram functions)
        fn (FaceNormalizer): An instance of a face normalizer used to align detected faces
        label (int): 0 for true humans and 1 for spoofs
        toFile (bool, optional): Can be set to false to disable output file creation
    
    Returns:
        list(list(int)): The calculated histograms
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
    """Loads and processes image lists using the :func:`data.loadFromList` funcion

    The output files will be called `<dataPrefix>_client_<dataSuffix>.txt` and `<dataPrefix>_imposter_<dataSuffix>.txt`

    Args:
        dataPath (str): The path to a directory where the output files will be created
        dataPrefix (str): Prefix for the output file names
        dataSuffix (str): Suffix for the output file names
        listPathTrue (str): Path the a file containg a list of relative paths to true images (labeled 0)
        listPathSpoof (str): Path the a file containg a list of relative paths to spoof images (labeled 1)
        histFunc (callable): A function used to calculate a histogram for a single detected face
        grayscale (bool): If true the images will be converted to grayscale before processing (that is required by some histogram functions)

    Returns:
        (list(list(int)), list(list(int))): The calculated histograms in the first column and their labels in the second in a randomized order
    """

    fn = FaceNormalizer()

    print("\tLoading client data...")
    data = loadFromList(listPathTrue, dataPath, dataPrefix + "_client_" + dataSuffix, histFunc, grayscale, fn, 0)

    print("\tLoading imposter data...")
    data.extend(loadFromList(listPathSpoof, dataPath, dataPrefix + "_imposter_" + dataSuffix, histFunc, grayscale, fn, 1))

    random.shuffle(data)
    return [d[0] for d in data], [d[1] for d in data]


def loadDataFile(filePath, label):
    """Loads a list of precalculated histograms from a text file

    Args:
        filePath (str): The path to the text file
        label (int): 0 for true humans and 1 for spoofs
    
    Returns:
        list(list(int)): The loaded histograms
    """

    hists = []
    file = open(filePath, "r")
    
    for hist in file:
        hists.append((eval(hist), label))

    return hists


def getTrainingDataFromFile(path, prefix, suffix):
    """Loads precalculated histograms from text files using :func:`data.loadDataFile`

    The files the will be loaded should be called `<prefix>_client_<suffix>.txt` and `<prefix>_imposter_<suffix>.txt`

    Args:
        path (str): The directory where the text files are located
        prefix (str): Prefix for the text file names
        suffix (str): Suffix for the text file names

    Returns:
        (list(list(int)), list(list(int))): The calculated histograms in the first column and their labels in the second in a randomized order
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