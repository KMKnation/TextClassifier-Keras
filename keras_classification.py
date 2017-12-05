import numpy as np
from skimage import io, color, exposure, transform
import glob, os
import h5py
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import  Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2

NUM_CLASSES = 36
IMG_SIZE = 28

#function to preprocess the image
def preprocess_img(img):
    try:
        # histogram normalization in y
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)
    except (ValueError):
        img = np.stack((img,) * 3)
        img = np.transpose(img, (1, 2, 0))
        pass


    # central crop
    # min_side = min(img.shape[:-1])
    # centre = img.shape[0]//2, img.shape[1]//2
    # img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
    #       centre[1] - min_side // 2:centre[1] + min_side // 2,:]

    #rescale to standard size
    # img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # cv2.imshow('img', img)
    # cv2.waitKey()

    #roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img


from  get_set_class_index import getAllIndexFromCharacter
def get_class(img_path):
    return int(getAllIndexFromCharacter(img_path.split('/')[-2]))

from get_set_class_index import getAllCharacterFromIndex
def get_class_name(index):
    return int(getAllCharacterFromIndex(index))


'''Load training images into numpy array'''
try:
    with h5py.File('X.h5', 'r') as hf:
        X, Y = hf['imgs'][:], hf['labels'][:]

        print('Length of X is {} and Length Of Y is {}'.format(len(X), len(Y)))
        print("Loaded images from X.h5")
except (IOError, OSError):
    print("Error occured while reading X.h5")

    root_dir = 'BillAddedData/training-images'

    imgs = []
    labels = []

    all_img_paths = glob.glob((os.path.join(root_dir, '*/*.png')))
    np.random.shuffle(all_img_paths)

    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if (len(imgs) % 1000 == 0): print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    print('Length of X is {} and Length Of Y is {}'.format(len(X), len(Y)))
    with h5py.File('X.h5', 'w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)


#convolutional nueral network
def cnn_model():
    model = Sequential()

    #1st layer
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    #2nd layer
    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    #3rd layer
    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    #flat
    model.add(Flatten())
    #512 is out dendries
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    #TOTAL CLASS OUT PUT DENDRIES
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model

model = cnn_model()

#lets train the model using SGD + momentum (how original)
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))


''' TRAINING '''

batch_size = 32
nb_epoch = 40

model.fit(X, Y,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_split=0.2,
          shuffle=True,
          callbacks=[LearningRateScheduler(lr_schedule),
                     ModelCheckpoint('model_some_bill_data.json.h5', save_best_only=True)])

# serialize model to JSON
model_json = model.to_json()
with open("model_some_bill_data.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("model.h5")
print("Saved model to disk")