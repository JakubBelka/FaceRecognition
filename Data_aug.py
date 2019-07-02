import cv2
import numpy as np
import os
from keras_preprocessing.image import img_to_array, ImageDataGenerator

#Pętla po zdjęciach w danym folderze:
for filename in os.listdir('/Users/kubab/PycharmProjects/ProjektInd/Domino/'):
    filename2=('/Users/kubab/PycharmProjects/ProjektInd/Domino/' + filename)

#OPERACJA NR 1 : BLUR.
    img = cv2.imread(filename2)
    rows, cols, ch = img.shape
#ZAKRES OZNACZA ZARÓWNO ILOŚĆ POWSTAŁYCH NOWYCH ZDJĘĆ
#JAK RÓWNIEŻ POZIOM WPROWADZANYCH ZNIEKSZTAŁCEŃ
    for i in range(3,10):
        blur = cv2.blur(img, (i*1, i*1))
        cv2.imwrite('/Users/kubab/PycharmProjects/ProjektInd/obrot/blur'+i.__str__()+filename, blur)

#INICJALIZACJA ZMIENNYCH POTRZEBNYCH DO DALSZYCH OPERACJI
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(brightness_range=[0.2, 1.7])
    datagenZ = ImageDataGenerator(zoom_range=[0.6, 1.1])
    it = datagen.flow(samples, batch_size=1)
    itZ = datagenZ.flow(samples, batch_size=1)
    datagenR = ImageDataGenerator(rotation_range=75)
    itR = datagenR.flow(samples, batch_size=1)

#pĘTLA WYKONUJĄCA OPERACJĘ NR 2 - ZMIANE JASNOŚCI
#ZAKRES OZNACZA ILOŚĆ POWSTAŁYCH ZDJĘĆ
    for i in range(8):
        batch = it.next()
        image = batch[0].astype('uint8')
        cv2.imwrite('/Users/kubab/PycharmProjects/ProjektInd/obrot/light' + i.__str__() + filename, image)

# PĘTLA WYKONUJĄCA OPERACJĘ NR 3 - ZOOM
# ZAKRES OZNACZA ILOŚĆ POWSTAŁYCH ZDJĘĆ
    for i in range(10):
        batchZ = itZ.next()
        # convert to unsigned integers for viewing
        imageZ = batchZ[0].astype('uint8')
        # plot raw pixel data
        cv2.imwrite('/Users/kubab/PycharmProjects/ProjektInd/obrot/zoom' + i.__str__() + filename, imageZ)

# PĘTLA WYKONUJĄCA OPERACJĘ NR 3 - ZOOM
# ZAKRES OZNACZA ILOŚĆ POWSTAŁYCH ZDJĘĆ
    for i in range(10):
        # generate batch of images
        batchR = itR.next()
        # convert to unsigned integers for viewing
        imageR = batchR[0].astype('uint8')
        cv2.imwrite('/Users/kubab/PycharmProjects/ProjektInd/obrot/rotC' + i.__str__() + filename, imageR)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
