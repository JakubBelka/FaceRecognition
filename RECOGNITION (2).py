import cv2
import numpy as np
import tensorflow as tf
import os

#Wczytanie modelu Caffe
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
cap = cv2.VideoCapture(0)
#Nadanie nazw dla klas
CATEGORIES = ["Domino  ", "Kuba  ", "Wojti  "]
path_output = 'C:/Users/kubab/PycharmProjects/ProjektInd/'
#Funkcja do obróbki obrazu na taki, któego spodziewa się sieć
def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    new_array = cv2.resize(gray_image, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#Wczytanie naszego modelu
model = tf.keras.models.load_model("model.model")

while (1):
    #DETEKCJA TWARZY
    _, frame = cap.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    #PĘTLA PO ODNALEZIONYCH TWARZACH
    for i in range(0, detections.shape[2]):
        # confidence
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue
        #OBRÓBKA OBRAZ - PRZYCIANANIE, FUNKCJA PREPARE O
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cropped = frame[startY - h:endY, startX - w:endX]
        cropped_photo = cv2.imwrite(os.path.join(path_output, 'elo.jpg'), cropped)
        gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # COLOR_BGR2GRAY(capture)
        #PREDYKCJA
        prediction = model.predict(prepare(cropped_photo))

        pewnosc = np.amax(prediction)
        pewnosc2 = pewnosc*100
        znak = '%'
        #JEŚLI NASZA SIEĆ JEST PEWNA NA 80% WYŚWIETLA NAZWĘ
        if pewnosc < 0.8:
            text = 'Unknown'
            text2 =' '
        elif pewnosc > 0.8:
            text = CATEGORIES[np.argmax(prediction)] + round(pewnosc2, 2).__str__()+znak
        #WYŚWIETLENIE WYNIKÓW PREDYKCJI NA EKRANIE
        print(prediction)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imshow('Obraz Rzeczywisty', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
