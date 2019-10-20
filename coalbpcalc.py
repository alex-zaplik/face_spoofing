import numpy as np
import cv2

class CoALBP:
    def __init__(self):
        self.deltasTimes = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        self.deltasPlus = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    def lbpOperator(self, img, x, y, w, h, deltas, r, extract=lambda c: c):
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
        resImg = rawImage.astype(np.uint8)
        (h, w) = rawImage.shape[:2]
        deltas = self.deltasTimes if useTimes else self.deltasPlus

        for x in range(0, w):
            for y in range(0, h):
                resImg[y, x] = self.lbpOperator(rawImage, x, y, w, h, deltas, r, extract=extract)

        return resImg
    
    def cooccurrenceMatrix(self, img, w, h, a, d):
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
        # TODO: Should all LBPs be computed or only a sparser grid?
        # TODO: Test this

        (h, w) = img.shape[:2]
        lbpImg = self.lbp(img, useTimes, r, extract=extract)

        vector = np.concatenate([
            self.cooccurrenceMatrix(lbpImg, w, h, a, d).flatten()
            for a in [(0, d), (d, 0), (d, d), (-d, d)]
        ])

        return list(map(int, vector))
