from lbpcalc import LBPCalc

import cv2


lbp = LBPCalc(((16, 2), (8, 2), (8, 1)))


def maattaHistogram(img):
    # Total histograms
    hist_16_2 = lbp.histogram(img, (16, 2))
    hist_8_2 = lbp.histogram(img, (8, 2))

    # 3x3 overlapping regions with a 14 pixel overlap
    localHists = []
    for xOffset in [-7, 14, 35]:
        for yOffset in [-7, 14, 35]:
            localHists += lbp.histogram(img, (8, 2), windowSize=(35, 35), xOffset=xOffset, yOffset=yOffset)
    
    return hist_16_2 + localHists + hist_8_2


def colorspaceHistogram(img, space='RGB'):
    hist = []
    if space == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif space == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif space == 'Dual':
        return colorspaceHistogram(img, space='YCrCb') + colorspaceHistogram(img, space='HSV')

    # Computing the HSV histogram
    hist  = lbp.histogram(img, (8, 1), extract=lambda c: c[0])
    hist += lbp.histogram(img, (8, 1), extract=lambda c: c[1])
    hist += lbp.histogram(img, (8, 1), extract=lambda c: c[2])

    return hist