import numpy as np
import cv2
from matplotlib import pyplot as plt


def isUniform(n, c):
    """A function checking whether a binary sequence is uniform or not

    A binary sequence is called uniform iff it cantains up to two binary
    transitions. For example: the sequnces 001100 and 111000 are uniform
    and 11001100 is not.

    Args:
        n (int): Length of the bit sequence
        c (int): Integer containing the bit sequnece as its suffix

    Returns:
        bool: True if the sequence is uniform
    """

    last = n % 2
    count = 0

    for _ in range(c):
        b = n % 2
        if last != b:
            if count > 2:
                break

            count += 1
            last = b
        n //= 2
    
    return count <= 2


class LBPCalc:
    """
    A general implementation of the LBP operator. Allows the use of
    different operator sizes and strides
    """

    def __init__(self, sizes):
        """
        Args:
            sizes (tuple((int, int))): LBP sizes to be precalculated for faster execution
        """

        self.sizes = sizes
        self.maps = dict()
        self.revMaps = dict()
        self.lbpDeltas = dict()

        self.precalcMaps()
        self.precalcLbpDeltas()
    

    def calcMap(self, p):
        """Calculates a dictionary that maps every uniform sequence to itself
        and every non-uniform one to 5 (the smallest number that is not uniform
        used as an extra label)

        Args:
            p (int): The length of the binary squences

        Returns:
            dict of int: int: The mapping that was calculated
        """

        mapping = dict()
        for i in range(2 ** p):
            if isUniform(i, p):
                mapping[i] = i
            else:
                mapping[i] = 5

        return mapping
    

    def calcRevMap(self, p):
        """Calculates a dictionary that maps uniform sequences (and the extra label 5)
        to a contiguous sequence of integers

        Args:
            p (int): The length of the binary squences

        Returns:
            dict of int: int: The mapping that was calculated
        """

        uniforms = [i for i in range(2 ** p) if isUniform(i, p)] + [5]

        mapping = dict()
        counter = 0
        for u in uniforms:
            mapping[u] = counter
            counter += 1

        return mapping
    

    def precalcMaps(self):
        """Prepares the precalulated maps
        """

        for size in self.sizes:
            self.maps[size] = self.calcMap(size[0])
            self.revMaps[size] = self.calcRevMap(size[0])
    

    def precalcLbpDeltas(self):
        """Calculates the relative positions of points used by the LBP operator
        in advance to optimaze the calculations
        """
        for size in self.sizes:
            deltas = []
            p = size[0]
            r = size[1]

            for j in range(p):   
                a = j * (2 * np.pi / p) 
                xx = int(np.round(r * np.cos(a)))
                yy = int(np.round(r * np.sin(a)))
                
                deltas.append((xx, yy))
            
            self.lbpDeltas[size] = deltas


    def lbpOperator(self, img, x, y, w, h, deltas, extract=lambda c: c, step=1):
        """Calculates the LBP label for a pixel at given coordinated using deltas
        to sample neighbouring pixels. This method is described in greater detail
        in the thesis

        Args:
            img (numpy.ndarray): The image to be provessed
            x (int): X coordinate of the center pixel
            y (int): Y coordinate of the center pixel
            w (int): Width of the image
            h (int): Height of the image
            deltas (list((int, int))): Pixel offsets for retiving samples
            extract (callable, optional): Method of extracting color values from a pixel
            step (int, optional): Step size along a single axis
        
        Returns:
            The calculated LBP descriptor
        """
        
        val = 0
        c = int(extract(img[y, x]))

        for j in range(len(deltas)):
            d = deltas[j]
            xx = (x + d[0] * step) % w
            yy = (y + d[1] * step) % h

            col = int(extract(img[yy, xx]))
            val += int(2 ** j) if col - c >= 0 else 0

        return val


    def lbp(self, rawImage, size, windowSize=(-1, -1), xOffset=0, yOffset=0, extract=lambda c: c, step=1):
        """The LBP calculator

        Described in detail in the thesis

        Args:
            rawImage (numpy.ndarray): The image to be processed
            size ((int, int)): The size of the operator
            windowSize ((int, int)): The size of the part of the image that will be used in computation given as (height, width)
            xOffset (int, optional): Starting pixel x offset
            yOffset (int, optional): Starting pixel y offset
            extract (callable, optional): Method of extracting color values from a pixel
            step (int, optional): Step size along a single axis

        Returns:
            numpy.ndarray: The processed 16-bit image
        """

        p = size[0]
        r = size[1]

        resImg = rawImage.astype(np.uint16)
        (h, w) = rawImage.shape[:2] if windowSize[0] == -1 and windowSize[1] == -1 else windowSize
        deltas = self.lbpDeltas[size]

        for x in range(xOffset, w, step):
            for y in range(yOffset, h, step):
                resImg[y, x] = self.lbpOperator(rawImage, x, y, w, h, deltas, extract=extract, step=step)

        return resImg
    

    def histogram(self, rawImage, size, windowSize=(-1, -1), xOffset=0, yOffset=0, extract=lambda c: c, step=1):
        """Computes the LBPs of an image and generates a histogram of the
        calculated data

        Args:
            rawImage (numpy.ndarray): The image to be processed
            size ((int, int)): The size of the LBP operator
            windowSize ((int, int)): The size of the part of the image that will be used in computation given as (height, width)
            xOffset (int, optional): Starting pixel x offset for the LBP operator
            yOffset (int, optional): Starting pixel y offset for the LBP operator
            extract (callable, optional): Method of extracting color values from a pixel
            step (int, optional): LBP operator step size along a single axis

        Returns:
            list(int): The calcualted histogram
        """

        img = self.lbp(rawImage.copy(), size, windowSize=windowSize, xOffset=xOffset, yOffset=yOffset, extract=extract, step=step)
        p = size[0]
        h, w = img.shape[:2]

        hist = [0 for _ in range(p * (p - 1) + 3)]
        mapping = self.maps[size]
        reverseMapping = self.revMaps[size]
        for x in range(w):
            for y in range(h):
                mapped = mapping[extract(img[y, x])]
                index = reverseMapping[mapped]
                
                hist[index] += 1

        # plt.bar([i for i in range(len(hist))], hist)
        # plt.show()

        return hist

