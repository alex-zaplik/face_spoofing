import cv2
import dlib
import imutils

from matplotlib import pyplot as plt
from imutils.face_utils import FaceAligner, rect_to_bb


class FaceNormalizer:

    def __init__(self, desiredFaceWidth=203, desiredLeftEye=(0.25, 0.25)):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.aligner = FaceAligner(self.predictor, desiredFaceWidth=desiredFaceWidth, desiredLeftEye=desiredLeftEye)
    
    def normalizedFaces(self, img, faceW=64, faceH=64, grayscale=True):
        faces = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 2)

        for rect in rects:
            x, y, h, w = rect_to_bb(rect)

            faceOriginal = img[y:y + h, x: x + w]
            faceAligned = self.aligner.align(img, gray, rect)
            
            if grayscale:
                faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)

            faces.append(imutils.resize(faceAligned, width=faceW, height=faceH))
        
        return faces