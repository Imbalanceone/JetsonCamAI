import numpy as np
import json
#import Object_detection_video as ODV

def CheckPredict(boxes, scores, classes, height, width):
    i=0
    while scores[0][i]>0.2:
        i+=1
    if i<0:
        return
    nboxes=boxes[0][:i][:]
    nscores=scores[0][:i]
    nclasses=classes[0][:i]
    #height=720
    #width=1280
    for i in nboxes:
        i[0]=int(i[0]*height)
        i[1]=int(i[1]*width)
        i[2]=int(i[2]*height)
        i[3]=int(i[3]*width)
    return nboxes, nscores, nclasses
    
def GetCoordinates(nboxes):
    #print(nboxes)
    #for i in nboxes:
        #print(i)
    ymin,xmin,ymax,xmax=nboxes
    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)
    return ymin, ymax, xmin, xmax


def GetJson(boxes, scores, classes):
    i = 0
    while scores[0][i] > 0.1:
        i+=1
    #i -= 1
    if i == 0:
        return "Nothing"
    nboxes = boxes[0][:i][:].copy()
    nscores = scores[0][:i].copy()
    nclasses = classes[0][:i].copy()
    nboxes = np.rot90(nboxes, 1)
    data = np.vstack((nclasses, nscores, nboxes))
    JsonArr = data.tolist()
    
    JsonArr = json.dumps(JsonArr)
    return JsonArr
