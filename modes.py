from methods import maattaHistogram, colorspaceHistogram, grayscaleLBP, grayscaleMultiCoALBP
from modelhandlers import tuneModel, trainModel, predict

import cv2
import data

def modeGen(args):
    dataPath = args.dataPath
    dataPrefix = args.dataPrefix
    useCoALBP = args.CoALBP

    methodDict = {
        "Maatta" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, maattaHistogram, grayscale=True
        ),
        "HSV" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, lambda img: colorspaceHistogram(img, space="HSV", useCoALBP=useCoALBP), grayscale=False
        ),
        "YCrCb" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, lambda img: colorspaceHistogram(img, space="YCrCb", useCoALBP=useCoALBP), grayscale=False
        ),
        "Dual" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, lambda img: colorspaceHistogram(img, space="Dual", useCoALBP=useCoALBP), grayscale=False
        ),
        "Gray" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, lambda img: grayscaleLBP(img, useCoALBP=useCoALBP), grayscale=True
        ),
        "GrayMulti" : lambda suffix, t, s: data.getTrainingData(
            dataPath, dataPrefix, suffix, t, s, grayscaleMultiCoALBP, grayscale=True
        )
    }

    if args.method is not None:
        if args.testTrue is None and args.testSpoof is None:
            prin("Processing data...")
            methodDict[args.method]("all", args.trainTrue, args.trainSpoof)
            print("Data processed")
        elif args.testTrue is None or args.testSpoof is None:
            print("Error: Only one test set given")
            return
        else:
            print("Processing data...")
            methodDict[args.method]("train", args.trainTrue, args.trainSpoof)
            methodDict[args.method]("test", args.testTrue, args.testSpoof)
            print("Data processed")
    else:
        print("Error: Method not given")
        return


def modeTune(args):
    # TODO: Loading with 4-fold CV (if _train/_test files don't exist, check _all)

    dataPath = args.dataPath
    dataPrefix = args.dataPrefix

    if dataPath is None or dataPrefix is None:
        missing = [i[1] for i in [(dataPath, "dataPath"), (dataPrefix, "dataPrefix")] if i[0] is None]
        print("Error: Not all data path parameters given. Misssing", missing)
        return
    
    if args.modelPath is not None and args.method is None:
        print("Error: Method not given")
        return

    print("Loading data...")
    xTrain, yTrain = data.getTrainingDataFromFile(dataPath, dataPrefix, "train")
    xTest, yTest = data.getTrainingDataFromFile(dataPath, dataPrefix, "test")
    print("Data loaded")

    clf, threshold, _ = tuneModel(xTrain, yTrain, xTest, yTest, kernel=args.kernel, loging=args.log)

    if args.modelPath is not None:
        print("Saving model to file...")
        data.saveClassifier(clf, threshold, args.method, args.modelPath)    
        print("Model saved")


def modeTrain(args):
    # TODO: Loading with 4-fold CV (if _train/_test files don't exist, check _all)

    dataPath = args.dataPath
    dataPrefix = args.dataPrefix

    C = eval(args.C)
    gamma = eval(args.Gamma)

    if dataPath is None or dataPrefix is None:
        missing = [i[1] for i in [(dataPath, "dataPath"), (dataPrefix, "dataPrefix")] if i[0] is None]
        print("Error: Not all data path parameters given. Misssing", missing)
        return

    if C is None or gamma is None:
        missing = [i[1] for i in [(C, "C"), (gamma, "Gamma")] if i[0] is None]
        print("Error: Not all model parameters given. Misssing", missing)
        return

    if args.modelPath is not None and args.method is None:
        print("Error: Method not given")
        return

    print("Loading data...")
    xTrain, yTrain = data.getTrainingDataFromFile(dataPath, dataPrefix, "train")
    xTest, yTest = data.getTrainingDataFromFile(dataPath, dataPrefix, "test")
    print("Data loaded")

    clf, threshold, spoofPredictor = trainModel(xTrain, yTrain, xTest, yTest, C=C, gamma=gamma, kernel=args.kernel, verbose=args.verbose)

    if args.modelPath is not None:
        print("Saving model to file...")
        data.saveClassifier(clf, threshold, args.method, args.modelPath)    
        print("Model saved")


def modeClass(args):
    methodDict = {
        "Maatta" : (maattaHistogram, True),
        "HSV" : (lambda img: colorspaceHistogram(img, space="HSV"), False),
        "YCrCb" : (lambda img: colorspaceHistogram(img, space="YCrCb"), False),
        "Dual" : (lambda img: colorspaceHistogram(img, space="Dual"), False),
        "Gray" : (grayscaleLBP, True),
        "GrayMulti" : (grayscaleMultiCoALBP, True)
    }

    dataPath = args.photoPath
    modelPath = args.modelPath

    if dataPath is None:
        print("Error: Photo path not given")

    if modelPath is None:
        print("Error: Model path not given")

    clf, threshold, method = data.loadClassifier(modelPath)
    X = data.loadPhoto(dataPath, methodDict[method][0], methodDict[method][1])
    y = predict(clf, X, threshold, probability=True)

    print(y)

    # This will be removed soon:
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(dataPath)
    for yy in y:
        c = (0, 0, 255) if yy[0] else (0, 255, 0)
        x, y, w, h = yy[1]
        
        text = "Imposter" if yy[0] else "Client"
        text += " %f" % (yy[2] * 100) + "%"

        cv2.rectangle(img, (x, y), (x + w, y + h), c, 3)
        cv2.putText(img, text, (x, y + h + 15), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y + h + 15), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Result", img)
    cv2.waitKey()
