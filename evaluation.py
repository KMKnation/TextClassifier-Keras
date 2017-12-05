import h5py
import numpy as np
from skimage import io, color, exposure, transform
import glob, os
from keras.models import Sequential, model_from_json

NUM_CLASSES = 36
IMG_SIZE = 28


from  get_set_class_index import getAllIndexFromCharacter
def get_class(img_path):
    return int(getAllIndexFromCharacter(img_path.split('/')[-2]))

from get_set_class_index import getAllCharacterFromIndex
def get_class_name(index):
    return int(getAllCharacterFromIndex(index))

#function to preprocess the image
def preprocess_img(img):
    #histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    #central crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2,:]

    #rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    #roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img


'''Load training images into numpy array'''
try:
    with h5py.File('X_test.h5', 'r') as hf:
        X, Y = hf['imgs'][:], hf['labels'][:]

        print('Length of X is {} and Length Of Y is {}'.format(len(X), len(Y)))
        print("Loaded images from X.h5")
except (IOError, OSError):
    print("Error occured while reading X_test.h5")

    root_dir = 'Data/test-images'

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
    with h5py.File('X_test.h5', 'w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

# model = Sequential()
#load json model

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights('model.h5')
from keras.models import load_model
model = load_model('models/pixelate models/96 accuracy/model_some_bill_data.json.h5')


y_pred = model.predict_classes(X)


import debug_tools

# debug_tools.plot_images(X[9:51], (IMG_SIZE, IMG_SIZE), Y[9:51], y_pred[9:51])

debug_tools.plot_confusion_matrix(Y, y_pred, NUM_CLASSES)

#converting into eye because we got classes index and our label is converted into matrix
y_pred = np.eye(NUM_CLASSES, dtype='uint8')[y_pred]

acc = np.sum(y_pred==Y)/np.size(y_pred)
print("Test accuracy = {}".format(acc))