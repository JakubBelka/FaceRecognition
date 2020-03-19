from keras.preprocessing.image import load_img
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout,  Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam


pic_size = 100
base_path = "C:/Users/kubab/PycharmProjects/ProjektInd/images/"
#PĘTLA PO ZDJECIACH KLAS W FOLDERZE TRAIN
for people in os.listdir(base_path + "train/"):
    for i in range(1,6):
        img = load_img(base_path + "train/" + people + "/" +os.listdir(base_path + "train/" + people)[i], target_size=(pic_size, pic_size))

for expression in os.listdir(base_path + "train"):
    print(str(len(os.listdir(base_path + "train/" + expression))) + " " + expression + " images")
#ZMIENNE
batch_size = 3
datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

#PRZYGOTOWANIE DANYCH TRAIN I VALIDATION
train_generator = datagen_train.flow_from_directory(base_path + "train",
                                                        target_size=(pic_size, pic_size),
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

validation_generator = datagen_validation.flow_from_directory(base_path + "validation",
                                                                  target_size=(pic_size, pic_size),
                                                                  color_mode="grayscale",
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical',
                                                                  shuffle=False)
#ARCHITEKTURA MODELU
nb_classes = 3

model = Sequential()
# 
model.add(Conv2D(512,(3,3), padding='same', input_shape=(100, 100,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
#
model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# 
model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# 4 
model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# 5 
model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# 6 
model.add(Conv2D(64,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# 7 
model.add(Conv2D(64,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# 8 
model.add(Conv2D(32,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
#9 
model.add(Conv2D(32,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# Flattening
model.add(Flatten())
# 
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# 
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
# 
model.add(Dense(nb_classes, activation='softmax'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#TRENOWANIE MODELU
epochs = 20
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=train_generator.n//train_generator.batch_size,
                                epochs=epochs,
                                validation_data = validation_generator,
                                validation_steps = validation_generator.n//validation_generator.batch_size,
                                callbacks=callbacks_list
                                )




model.save('MODEL.model')
