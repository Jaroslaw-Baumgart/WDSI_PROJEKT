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

def learn_bovw(data):
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)
    sift = cv2.SIFT_create()
    for image in data:
        image['image'] = cv2.cvtColor(image['image'], cv2.COLOR_BGR2GRAY)
    for image in data:
        kpoints = sift.detect(image['image'], None)
        _, desc = sift.compute(image['image'], kpoints)
        if desc is not None:
            bow.add(desc)
    vocabulary = bow.cluster()
    np.save('voc.npy', vocabulary)

def extract_features(data):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for image in data:
        kpoints = sift.detect(image['image'], None)
        imgDes = bow.compute(image['image'], kpoints)
        if imgDes is not None:
            image.update({'desc': imgDes})
        else:
            image.update({'desc': np.zeros((1, 128))})
    return data

def main():
    train_data = load_data_train()
    test_data = load_data_test()
    if os.path.isfile('voc.npy') != True:
        learn_bovw(train_data)
    train_data = extract_features(train_data)
    rf = train(train_data)
    com = input()
    if com == 'classify':
        n = int(input())
        images = [[] for i in range(n)]
        for i in range(n):
            images[i] = ({'n' : 0, 'xy' : [], 'name' : ''})
        for i in images:
            i['name'] = input()
            i['n'] = int(input())
            i['xy'] = [[] for i in range(i['n'])]
            for j in range(i['n']):
                    i['xy'][j].append(input())
    images_data = []
    for i in images:
        for image in test_data:
            if image['name'] == i['name']:
                for j in range(i['n']):
                    xmin = int(i['xy'][j][0].split(sep=' ')[0])
                    xmax = int(i['xy'][j][0].split(sep=' ')[1])
                    ymin = int(i['xy'][j][0].split(sep=' ')[2])
                    ymax = int(i['xy'][j][0].split(sep=' ')[3])
                    images_data.append({'image' : image['image'][ymin:ymax, xmin:xmax]})
    images_data = extract_features(images_data)
    images_data = predict(rf,images_data)
    for i in images_data:
        if i['ID_pred'] == 1:
            print('speedlimit')
        else:
            print('other')
    return

if __name__ == '__main__':
    main()