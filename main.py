import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as ET
import pandas
import os


def load_data_train(path):
    dane = []

    for a in os.listdir('../train/images'):
        file=os.path.join('../train/images',a) #name, type, ilość, kordynaty
        name=os.path.basename(file)
        name=name.split(sep='.')
        name=name + '.xml'
        xmlfile = os.path.join('../train/annotations',name)
        tree= ET.parse(xmlfile)
        root=tree.getroot()
        name=root[1]

        iter = 4
        while iter is not len(root):
            xmin=int(root[iter][5][0].text)
            ymin=int(root[iter][5][1].text)
            xmax=int(root[iter][5][2].text)
            ymax=int(root[iter][5][3].text)
            classID=root[iter][0].text
            image=cv2.imread(file,cv2.IMREAD_COLOR)
            if classID=='speedlimit' :
                ID=1
            else:
                ID=0
            dane=({'image':image[ymin:ymax,xmin:xmax],'name':name,'ID':ID})
            iter = iter + 1

    return dane


def load_data_test(path):
    dane = []

    for a in os.listdir('../test/images'):
        file = os.path.join('../test/images', a)
        name = os.path.basename(file)
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        dane.append({'image':image,'name':name})

    return dane



