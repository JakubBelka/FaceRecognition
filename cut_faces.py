import cv2
import numpy as np
import os


#SIEC WYSZUKUJACA TWARZE
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")


#PETLA PO PLIKACH W FOLDERZE "ZDJ". W NIM ZNAJDUJA SIE ZDJECIA
#z KTÓRYCH WYCIETE ZOSTANA TWARZE I UMIESZCZINE W BAZIE DANYCH
for filename in os.listdir('/Users/kubab/PycharmProjects/ProjektInd/ZDJ/'):
    filename2=('/Users/kubab/PycharmProjects/ProjektInd/it/' + filename)
#ZAPIS ZDJEC DO IMG A NASTEPNIE DO BLOB
    img = cv2.imread(filename2)
    image = cv2.resize(img, (300, 300))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

#DETEKCJA TWARZY
    net.setInput(blob)
    detections = net.forward()
#PĘTLA PO ODNALEZIONYCH TWARZACH
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
#JEŚLI SIEĆ JEST PEWNA NA MIN. 50% TWARZ ZOSTAJE WYCIĘTA
        if confidence > 0.5:
#WYCIĘCIE TWARZY
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cropped = image[startY - w:endY, startX - h:endX]

    # ZAPIS WYCIETEGO ZDJEĘCIA
    cv2.imwrite(filename, cropped)
    cv2.waitKey(0)

