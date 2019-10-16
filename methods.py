from lbpcalc import LBPCalc

import cv2


# LBP types that are used by this application represented as (p, q) pairs
lbp = LBPCalc(((16, 2), (8, 2), (8, 1)))


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


def grayscaleLBP(img):
    """The simples algorithm collecting a single :math:`LBP_{8, 1}` pass on a grayscale
    image into a histogram

    Args:
        img (numpy.ndarray): The image to be processed
    
    Returns:
        list(int): The calculated histogram
    """

    return lbp.histogram(img, (8, 1))


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


def colorspaceHistogram(img, space='RGB'):
    """A method proposed by TODO which uses three :math:`LBP_{8, 1}` passes, one over each channel
    of the given color space

    Args:
        img (numpy.ndarray): The image to be processed
        space (str, optional): The color space to be used. Options are 'HSV', 'YCrCb' and
        'Dual' (concatenation of 'HSV' and 'YCrCb'). If a different value is given the RGB
        color space will be used
    
    Returns:
        list(int): The concatination of the three calculated histograms
    """

    # TODO: Allow use of CoALBP

    hist = []
    if space == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif space == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif space == 'Dual':
        return colorspaceHistogram(img, space='YCrCb') + colorspaceHistogram(img, space='HSV')

    # Computing and concatenating histograms
    hist  = lbp.histogram(img, (8, 1), extract=lambda c: c[0])
    hist += lbp.histogram(img, (8, 1), extract=lambda c: c[1])
    hist += lbp.histogram(img, (8, 1), extract=lambda c: c[2])

    return hist