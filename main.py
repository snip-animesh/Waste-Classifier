# import necessary packages

import cvzone
from cvzone.ClassificationModule import Classifier
import cv2
import os

cap = cv2.VideoCapture(0)

# Defining Classifier
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)

# Import all the waste images
imgWastes = []
pathWaste = "Resources/Waste"
# make a list of all wastes
pathLst = os.listdir(pathWaste)
#  Now bring the actual image
for path in pathLst:
    imgWastes.append(cv2.imread(os.path.join(pathWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the bins images
imgBins = []
pathBins = "Resources/Bins"
# make a list of all wastes
pathLst = os.listdir(pathBins)
#  Now bring the actual image
for path in pathLst:
    imgBins.append(cv2.imread(os.path.join(pathBins, path), cv2.IMREAD_UNCHANGED))
# print(imgBins)
# waste number : bin number
# 0 = Recyclable , 1 = Hazardous , 2 = Food , 3 = REsidula
classDict = {
    0: None,
    1 : 0,
    2: 0,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
}

while True:
    succ, img = cap.read()
    # Resize the image to set it on background image
    imgResize = cv2.resize(img, (454, 340))

    # Making Prediction
    prediction = classifier.getPrediction(img)
    # print(prediction)
    # importing the background image
    imgBackground = cv2.imread('Resources/backgrounde.png')

    classID = prediction[1]
    if classID:
        #     # Checking if imgWaste works just overlaying a dummy image
        imgBackground = cvzone.overlayPNG(imgBackground, imgWastes[classID - 1], (900, 150))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (970, 340))

        classIDbin = classDict[classID]
        imgBackground = cvzone.overlayPNG(imgBackground, imgBins[classIDbin], (890, 394))

    # overlays
    imgBackground[148:148 + 340, 150:150 + 454] = imgResize  # height , width
    # cv2.imshow("Img",img)
    cv2.imshow("Background", imgBackground)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
cv2.destroyAllWindows()