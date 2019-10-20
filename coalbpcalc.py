import numpy as np
import cv2

class CoALBP:
    """
    A basic implementation of the CoALBP operator. Allows the use of
    the 'plus' and 'times' LBP operators
    """

    def __init__(self):
        self.deltasTimes = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        self.deltasPlus = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    def lbpOperator(self, img, x, y, w, h, deltas, r, extract=lambda c: c):
        """A simple LBP operator allowing only the plus and times variants

        Args:
            img (numpy.ndarray): The image to be provessed
            x (int): X coordinate of the center pixel
            y (int): Y coordinate of the center pixel
            w (int): Width of the image
            h (int): Height of the image
            deltas (list((int, int))): Pixel offsets for retiving samples
            r (int): The size of the operator
            extract (callable, optional): Method of extracting color values from a pixel
        
        Returns:
            The calculated LBP descriptor
        """

        val = 0
        c = int(extract(img[y, x]))

        for j in range(len(deltas)):
            d = deltas[j]
            xx = (x + d[0] * r) % w
            yy = (y + d[1] * r) % h

            col = int(extract(img[yy, xx]))
            val += int(2 ** j) if col - c >= 0 else 0
        
        return val
    
    def lbp(self, rawImage, useTimes, r, extract=lambda c: c):
        """The LBP calculator

        Described in detail in the thesis

        Args:
            rawImage (numpy.ndarray): The image to be processed
            useTimes (bool): If set the times operator will be used, plus otherwise
            r (int): The size of the operator
            extract (callable, optional): Method of extracting color values from a pixel

        Returns:
            numpy.ndarray: The processed 8-bit image
        """
        resImg = rawImage.astype(np.uint8)
        (h, w) = rawImage.shape[:2]
        deltas = self.deltasTimes if useTimes else self.deltasPlus

        for x in range(0, w):
            for y in range(0, h):
                resImg[y, x] = self.lbpOperator(rawImage, x, y, w, h, deltas, r, extract=extract)

        return resImg
    
    def cooccurrenceMatrix(self, img, w, h, a, d):
        """Calculates the co-occurrence matrix for the LBP descriptors

        Args:
            img (numpy.ndarray): The LBP image to be processed
            w (int): Width of the image
            h (int): Height of the image
            a ((int, int)): The displacements vector
            d (int): The length of the displacement vector
        
        Returns:
            numpy.ndarray: The co-occurrence matrix
        """

        mat = np.zeros((16, 16))

        for x in range(0, w):
            for y in range(0, h):
                xx = (x + a[0] * d) % w
                yy = (y + a[1] * d) % h

                c0 = img[y, x]
                c1 = img[yy, xx]

                mat[c0, c1] += 1
        
        return mat
        

    def feature(self, img, r, d, useTimes=False, extract=lambda c: c):
        """Calculates the co-occurrence matricies, flattens and concatinates
        them into a single feature vector.

        Args:
            img (numpy.ndarray): The LBP image to be processed
            r (int): The size of the LBP operator
            d (int): The length of the displacement vector
            useTime (bool, optional): If set the times operator will be used, plus otherwise
            extract (callable, optional): Method of extracting color values from a pixel
        
        Returns:
            list(int): The feature vector
        """

        # TODO: Should all LBPs be computed or only a sparser grid?
        # TODO: Test this

        (h, w) = img.shape[:2]
        lbpImg = self.lbp(img, useTimes, r, extract=extract)

        vector = np.concatenate([
            self.cooccurrenceMatrix(lbpImg, w, h, a, d).flatten()
            for a in [(0, d), (d, 0), (d, d), (-d, d)]
        ])

        return list(map(int, vector))
