from arguments import args
from modes import modeGen, modeTune, modeTrain, modeClass


# TODO: Get access to databeses
# TODO: Figure out a third method

modeDict = {
    "Gen": modeGen,
    "Tune": modeTune,
    "Train": modeTrain,
    "Class": modeClass
}

modeDict[args.mode](args)

# 16-36 -> YCrCb

# python .\terminalEntry.py -mo Gen -me Gray -dpa "out/gray/" -dpr "from_raw" -trt "raw/client_train_raw.txt" -trs "raw/imposter_train_raw.txt" -tet "raw/client_test_raw.txt" -tes "raw/imposter_test_raw.txt"
# python .\terminalEntry.py -mo Tune -k rbf -me Gray -dpa "out/gray/" -mp "clfs/gray_lbp.svm" -dpr "from_raw" -l
# python .\terminalEntry.py -mo Train -k rbf -me Gray -dpa "out/gray/" -mp "clfs/gray_lbp.svm" -dpr "from_raw" -c "3" -g "5e-9" -v
# python .\terminalEntry.py -mo Class -mp "clfs/gray_lbp.svm" -pp "Images/0007_01_00_01_161.jpg"
