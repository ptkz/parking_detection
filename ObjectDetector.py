import cv2 as cv
import numpy

classNames = {0: 'occupied', 1: 'empty'}

class Detector:
    def __init__(self):
      global cvNet
      cvNet = cv.dnn.readNetFromTensorflow('model/frozen_inference_graph.pb',
                                            'model/graph.pbtxt')

    def detectObject(self, imName):
        img = cv.cvtColor(numpy.array(imName), cv.COLOR_BGR2RGB)
        cvNet.setInput(cv.dnn.blobFromImage(img, swapRB=True, crop=False))
        networkOutput = cvNet.forward()
        
        rows, cols, channels = img.shape
        
        for detection in networkOutput[0,0]:

            score = float(detection[2])
            if score > 0.8:
                class_id = int(detection[1])
                
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows

                if(class_id==0):
                    cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=1)
                if(class_id==1):
                    cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=1)
                
                # label = classNames[class_id] + ": " + str("{0:.2f}".format(score))
                # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_TRIPLEX, 0.5, 1)
                # yLeftBottom = max(top, labelSize[1])
                # cv.putText(img, label, (int(left+5), int(top)), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))                  
        
        img = cv.imencode('.jpg', img)[1].tobytes()
        return img
