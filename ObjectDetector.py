import os
import cv2
import numpy as np
import tensorflow as tf
import sys

classNames = {0: 'occupied', 1: 'empty'}

class Detector:
    def __init__(self):
        PATH_TO_MODEL = 'model/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)
   
    # def __init__(self):
    #   global cvNet
    #   cvNet = cv.dnn.readNetFromTensorflow('model/frozen_inference_graph.pb',
    #                                         'model/graph.pbtxt')

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})

            # print(boxes, scores, classes, num)
        return boxes, scores, classes, num

    # def detectObject(self, imName):
    #     img = cv.cvtColor(numpy.array(imName), cv.COLOR_BGR2RGB)
    #     cvNet.setInput(cv.dnn.blobFromImage(img, swapRB=True, crop=False))
    #     networkOutput = cvNet.forward()
        
    #     rows, cols, channels = img.shape
        
    #     for detection in networkOutput[0,0]:

    #         score = float(detection[2])
    #         if score > 0.8:
    #             class_id = int(detection[1])
                
    #             left = detection[3] * cols
    #             top = detection[4] * rows
    #             right = detection[5] * cols
    #             bottom = detection[6] * rows

    #             if(class_id==0):
    #                 cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=1)
    #             if(class_id==1):
    #                 cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=1)
                
    #             # label = classNames[class_id] + ": " + str("{0:.2f}".format(score))
    #             # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_TRIPLEX, 0.5, 1)
    #             # yLeftBottom = max(top, labelSize[1])
    #             # cv.putText(img, label, (int(left+5), int(top)), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))                  
        
    #     img = cv.imencode('.jpg', img)[1].tobytes()
    #     return img
