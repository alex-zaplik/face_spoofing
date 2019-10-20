import argparse


# Modes:
# - Gen -   Generates new trainting data
#           Args: method, dataPath, dataPrefix, trainTrue, trainSpoof [, testTrue, testSpoof]
# - Tune -  Tunes a model on the specified data
#           Args: method, dataPath, dataPrefix, modelPath
# - Train - Tunes a model on the specified data with given parameters
#           Args: method, dataPath, dataPrefix, modelPath, C, gamma
# - Class - Performs classification with a model and a photo
#           Args: modelPath, photoPath


parser = argparse.ArgumentParser()

parser.add_argument("--mode", "-mo", default="Class", choices=["Gen", "Tune", "Train", "Class"], required=True, help="The mode in which the application will be run. Default - Class")
parser.add_argument("--method", "-me", choices=["Maatta", "HSV", "YCrCb", "Gray", "Dual", "GrayMulti"], help="Method used for generating feauture vectors. Always required in Gen. Required in Tune and Train if modelPath was given")

parser.add_argument("--dataPath", "-dpa", help="In Gen, Tune and Train: The path for text files with calculated feature vectors (output in case of Gen, input otherwise). The files will be named: <dataPrefix>_(client|imposter)_(test|train).txt")
parser.add_argument("--dataPrefix", "-dpr", help="The file name prefix (see dataPath)")

parser.add_argument("--trainTrue", "-trt", help="Path to a text file containing a list of relative paths to photos of real people. Used for training the model. Required in Gen")
parser.add_argument("--trainSpoof", "-trs", help="Path to a text file containing a list of relative paths to photos of spoof attacks. Used for training the model. Required in Gen")
parser.add_argument("--testTrue", "-tet", help="Path to a text file containing a list of relative paths to photos of real people. Used for testing the model. Required in Gen")
parser.add_argument("--testSpoof", "-tes", help="Path to a text file containing a list of relative paths to photos of spoof attacks. Used for testing the model. Required in Gen")

parser.add_argument("--modelPath", "-mp", help="A path to a classifier file. Required in Class (input). Optional in Tune and Train if the trained model is to be saved (output)")
parser.add_argument("--photoPath", "-pp", help="A path of a single photo to be classified. Required in Class")

parser.add_argument("--kernel", "-k", choices=["rbf", "linear"], default="rbf", help="Kernel used for training an SVM. Optional in Tune and Train. Default - rbf")
parser.add_argument("--C", "-c", help="C parameter to be used for training an SVM. Required in Train")
parser.add_argument("--Gamma", "-g", help="Gamma parameter to be used for training an SVM. Required in Train. Ignored if the kernel is set to linear")
parser.add_argument("--CoALBP", "-co",  action="store_true", help="In Gen. If set CoALBP will be used instead of a standard LBP for feature generation when using the following methods: Gray, YCrCb, HSV, Dual")

parser.add_argument("--verbose", "-v", action="store_true", help="If set. More information will be printed to the console")
parser.add_argument("--log", "-l", action="store_true", help="In Gen. If set a log of the tuning process will be saved to a log file in ./logs")

args = parser.parse_args()