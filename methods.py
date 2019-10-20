from lbpcalc import LBPCalc
from coalbpcalc import CoALBP

import cv2


# LBP types that are used by this application represented as (p, q) pairs
lbp = LBPCalc(((16, 2), (8, 2), (8, 1)))
coalbp = CoALBP()


# Just because I'm curious
def curiousMethod(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hist = []

    for channel in [0, 1, 2]:
        for xOffset in [-7, 14, 35]:
            for yOffset in [-7, 14, 35]:
                hist += lbp.histogram(
                    img, (8, 2), windowSize=(35, 35),
                    xOffset=xOffset, yOffset=yOffset,
                    extract=lambda c: c[channel]
                )
    
    return hist


def grayscaleLBP(img, useCoALBP=False):
    """The simples algorithm collecting a single LBP pass on a grayscale
    image into a histogram

    Args:
        img (numpy.ndarray): The image to be processed
        useCoALBP (bool): If true the CoALBP descriptor is used otherwise :math:`LBP_{8, 1}`
    
    Returns:
        list(int): The calculated histogram
    """

    if useCoALBP:
        return coalbp.feature(img, 1, 2)

    return lbp.histogram(img, (8, 1))


def grayscaleMultiCoALBP(img):
    hist  = coalbp.feature(img, 1, 2)
    hist += coalbp.feature(img, 2, 4)
    hist += coalbp.feature(img, 4, 8)

    return hist


def maattaHistogram(img):
    """A method proposed by Maatta et al. which uses eleven passes concatenated
    to a single histogram. The passes are: An :math:`LBP_{16, 2}` and an :math:`LBP_{8, 2}` over the
    entire image and nine :math:`LBP_{8, 2}` passed over overlapping squares

    Args:
        img (numpy.ndarray): The image to be processed
    
    Returns:
        list(int): The concatination of the eleven calculated histograms
    """

    # Total histograms
    hist_16_2 = lbp.histogram(img, (16, 2))
    hist_8_2 = lbp.histogram(img, (8, 2))

    # 3x3 overlapping regions with a 14 pixel overlap
    localHists = []
    for xOffset in [-7, 14, 35]:
        for yOffset in [-7, 14, 35]:
            localHists += lbp.histogram(img, (8, 2), windowSize=(35, 35), xOffset=xOffset, yOffset=yOffset)
    
    return hist_16_2 + localHists + hist_8_2


def colorspaceHistogram(img, space='RGB', useCoALBP=False):
    """A method proposed by TODO which uses three LBP passes, one over each channel
    of the given color space

    Args:
        img (numpy.ndarray): The image to be processed
        space (str, optional): The color space to be used. Options are 'HSV', 'YCrCb' and
        'Dual' (concatenation of 'HSV' and 'YCrCb'). If a different value is given the RGB
        color space will be used
        useCoALBP (bool): If true the CoALBP descriptor is used otherwise :math:`LBP_{8, 1}`
    
    Returns:
        list(int): The concatination of the three calculated histograms
    """

    hist = []
    if space == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif space == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif space == 'Dual':
        return colorspaceHistogram(img, space='YCrCb') + colorspaceHistogram(img, space='HSV')

    # Computing and concatenating histograms
    if not useCoALBP:
        hist  = lbp.histogram(img, (8, 1), extract=lambda c: c[0])
        hist += lbp.histogram(img, (8, 1), extract=lambda c: c[1])
        hist += lbp.histogram(img, (8, 1), extract=lambda c: c[2])
    else:
        hist  = coalbp.feature(img, 1, 2, extract=lambda c: c[0])
        hist += coalbp.feature(img, 2, 4, extract=lambda c: c[1])
        hist += coalbp.feature(img, 4, 8, extract=lambda c: c[2])

    return hist