from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from lbpcalc import LBPCalc
import data


lbp = LBPCalc(((16, 2), (8, 2)))


def maattaHistogram(img):
    # Total histograms
    hist_16_2 = lbp.histogram(img, (16, 2))
    hist_8_2 = lbp.histogram(img, (8, 2))

    # Assuming image size is 64x64
    localHists = []
    size = 25    
    for x in [0, 19, 39]:
        for y in [0, 19, 39]:
            localHists += lbp.histogram(img[x:x + size, y:y + size], (8, 2))
    
    return hist_16_2 + localHists + hist_8_2


# Generating the histograms and savaing them in files
# print("Processing data...")
# data.getTrainingData("NormalizedFace", "train", "out2", maattaHistogram)
# data.getTrainingData("NormalizedFace", "test", "out2", maattaHistogram)
# print("Data processed")

print("Loading data...")
X_train, y_train = data.getTrainingDataFromFile("out2", "train")
X_test, y_test = data.getTrainingDataFromFile("out2", "test")
print("Data loaded")

print("Fitting the model...")
# TODO: Try to fit this
clf = svm.SVC(kernel='rbf', C=1e6, gamma=8e-12, verbose=True)
clf.fit(X_train, y_train)
print("Model fitted")

print("Testing...")
y_pred = clf.predict(X_test)
score = metrics.precision_score(y_test, y_pred)
print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, y_pred)))
print("Score=%f" % (score))