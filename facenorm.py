import cv2
import dlib
import imutils

from matplotlib import pyplot as plt
from imutils.face_utils import FaceAligner, rect_to_bb


class FaceNormalizer:
    """
    This class uses imutils.face_utils.FaceAligner to normalize a square photo
    of a face. This is used before feature extraction to achive more reliable
    results
    """

    def __init__(self, desiredFaceWidth=203, desiredLeftEye=(0.25, 0.25)):
        """
        Args:
            desiredFaceWidth (int, optional): desiredFaceWidth parameter of FaceAligner
            desiredLeftEye ((float, float), optional): desiredLeftEye parameter of FaceAligner
        """

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.aligner = FaceAligner(self.predictor, desiredFaceWidth=desiredFaceWidth, desiredLeftEye=desiredLeftEye)
    
    def normalizedFaces(self, img, faceW=64, faceH=64, grayscale=True, returnBox=False):
        """Returnes a list of normalized faces detected in the image

        Args:
            img (numby.ndarray): The image to be processed
            faceW (int, optional): The width of the resulting image
            faceH (int, optional): The height of the resulting image
            grayscale (bool, optional): If true the images will be converted to grayscale
            returnBox (bool, optional): If true the return type will change (see return types)
        
        Returns:
            list((numpy.ndarray, (int, int, int, int))): A list of detected and normalized
            faces. If 'returnBox' was set tu true the list contains pairs with an ndarray
            image of the face and a rectangle defined by the x, y coordinated of the top-left
            corner, the width and the height. If 'returnBox' was set to false just a list of
            ndarrays is returned
        """


        faces = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 2)

        for rect in rects:
            x, y, h, w = rect_to_bb(rect)

            faceOriginal = img[y:y + h, x: x + w]
            faceAligned = self.aligner.align(img, gray, rect)
            
            if grayscale:
                faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)

            if returnBox:
                faces.append((imutils.resize(faceAligned, width=faceW, height=faceH), (x, y, w, h)))
            else:
                faces.append(imutils.resize(faceAligned, width=faceW, height=faceH))
        
        return faces