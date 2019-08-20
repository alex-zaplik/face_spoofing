from lbpcalc import LBPCalc, isUniform
from matplotlib import pyplot as plt

import numpy as np

def genImages(p, r):
    deltas = []

    for j in range(p):   
        a = j * (2 * np.pi / p) 
        xx = int(np.round(r * np.cos(a)))
        yy = int(np.round(r * np.sin(a)))
        
        deltas.append((xx, yy, j))

    for i in range(2 ** p):
        img = np.zeros((1 + 2 * r, 1 + 2 * r), np.uint8)
        img[r, r] = 150

        for d in deltas:
            if i & (1 << d[2]):
                img[r + d[1], r + d[0]] = 255
        
        yield (img, i, isUniform(p, i))


def testLBP():
    lbp = LBPCalc(((16, 2), (8, 2), (8, 1)))
    
    print("Testing LBP_8_1")
    for img in genImages(8, 1):
        val_8_1 = lbp.lbpOperator(img[0], 1, 1, 3, 3, lbp.lbpDeltas[(8, 1)])
        
        if img[1] != val_8_1 or img[2] != isUniform(8, val_8_1):
            print("Error for LBP_8_2", img[0], img[1], val_8_1)

    print("Testing LBP_8_2")
    for img in genImages(8, 2):
        val_8_2 = lbp.lbpOperator(img[0], 2, 2, 5, 5, lbp.lbpDeltas[(8, 2)])
        
        if img[1] != val_8_2 or img[2] != isUniform(8, val_8_2):
            val_16_2 = lbp.lbpOperator(img[0], 2, 2, 5, 5, lbp.lbpDeltas[(16, 2)])
            print("Error for LBP_8_2", img[0], img[1], val_16_2, val_8_2)

    print("Testing LBP_16_2")
    for img in genImages(16, 2):
        val_16_2 = lbp.lbpOperator(img[0], 2, 2, 5, 5, lbp.lbpDeltas[(16, 2)])

        if img[1] != val_16_2 or img[2] != isUniform(16, val_16_2):
            val_8_2 = lbp.lbpOperator(img[0], 2, 2, 5, 5, lbp.lbpDeltas[(8, 2)])
            print("Error for LBP_16_2", img, val_16_2, val_8_2)

# for img in genImages(8):
#     plt.imshow(img[0])
#     plt.show()

testLBP()