import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as ET
import pandas
import os

# TODO Jakość kodu i raport (4/4)
# TODO Raport troche skapy.

# TODO Skuteczność klasyfikacji 0.929 (4/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0


# TODO Skuteczność detekcji (/2)

#wczytanie obrazów do trenowania oraz wydobycie informacji z plików xml
def load_data_train():
    dane = []
    for a in os.listdir('../train/images'):
        file = os.path.join('../train/images', a)
        name = os.path.basename(file)
        name = name.split(sep='.')
        name = name[0] + '.xml'
        xmlfile = os.path.join('../train/annotations', name)
        tree = ET.parse(xmlfile)
        root =tree.getroot()
        name = root[1]
        iter = 4
        # TODO Latwiej uzyc metody "find" i "findall".
        while iter is not len(root):
            x_min = int(root[iter][5][0].text)
            y_min = int(root[iter][5][1].text)
            x_max = int(root[iter][5][2].text)
            y_max = int(root[iter][5][3].text)
            classID = root[iter][0].text
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            if classID == 'speedlimit' :
                ID = 1
            else:
                ID = 0
            dane.append({'image' : image[y_min:y_max, x_min:x_max], 'name' : name, 'ID' : ID})
            iter = iter + 1
    return dane

#wczytanie obrazów testowanych
def load_data_test():
    dane = []
    for a in os.listdir('../test/images'):
        file = os.path.join('../test/images', a)
        name = os.path.basename(file)
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        dane.append({'image' : image, 'name' : name})
    return dane

#stworzenie słownika
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

#wyznaczenie deskryptorów obrazów
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

#stworzenie random forest
def train(data):
    rf = RandomForestClassifier(128)
    x_matrix = np.empty((1, 128))
    y_vector = []
    for image in data:
        y_vector.append(image['ID'])
        x_matrix = np.vstack((x_matrix, image['desc']))
    rf.fit(x_matrix[1:], y_vector)
    return rf

#funckja określająca ID danych podanych w argumencie poprzez kwalifikację
def predict(rf, data):
    for sample in data:
        sample.update({'ID_pred': rf.predict(sample['desc'])[0]})
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