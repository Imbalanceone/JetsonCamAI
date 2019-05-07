import numpy as np
import json
from config import Debug as D
from config import Score as S
#import Object_detection_video as ODV

def CheckPredict(boxes, scores, classes, height, width):
    i = 0
    while scores[0][i] > S:
        i += 1
    if i == 0:
        if D > 0:
            print("Nothing in CheckPredict")
        return
    nboxes = boxes[0][:i][:]
    nscores = scores[0][:i]
    nclasses = classes[0][:i]
    for i in nboxes:
        i[0] = int(i[0]*height)
        i[1] = int(i[1]*width)
        i[2] = int(i[2]*height)
        i[3] = int(i[3]*width)
    if D > 0:
        print(nboxes, nscores, nclasses)
    return nboxes, nscores, nclasses
    
def GetCoordinates(nboxes):
    if D > 0:
        print(nboxes)
    ymin,xmin,ymax,xmax = nboxes
    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)
    if D > 0:
        print(ymin, ymax, xmin, xmax)
    return ymin, ymax, xmin, xmax

def GetJson(boxes, scores, classes):
    i = 0
    while scores[0][i] > S:
        i += 1
    if i == 0:
        if D > 0:
            print("Nothing in frame")
        return "Nothing"
    nboxes = boxes[0][:i][:].copy()
    nscores = scores[0][:i].copy()
    nclasses = classes[0][:i].copy()
    nboxes = np.rot90(nboxes, 1)
    data = np.vstack((nclasses, nscores, nboxes))
    JsonArr = data.tolist()
    JsonArr = json.dumps(JsonArr)
    if D > 0:
        print(JsonArr)
    return JsonArr
