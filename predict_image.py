from keras.models import model_from_json
from skimage import io, color, exposure, transform
import cv2
import numpy as np

IMG_SIZE = 28

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

    cv2.imshow('processed', img)

    #roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img

# json_file = open('model_some_bill_data.json.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights('model_some_bill_data.json.h5')
from keras.models import load_model
model = load_model('model_some_bill_data.json.h5')

image = cv2.imread('test/7.png')
cv2.imshow("original", image)
image = preprocess_img(image)

# cv2.imshow('processed', image)

'''
You have to reshape the input image to have a shape of [?, 3, 32, 32] where ? is the batch size. In your case, since you have 1 image the batch size is 1, so you can do:
'''
# img = np.array(img)
img = image.reshape((1, 3, 28, 28))
y_pred = model.predict_classes(img)

from get_set_class_index import getAllCharacterFromIndex
print(str(getAllCharacterFromIndex(y_pred[0])))

if(cv2.waitKey() == 0):
    cv2.destroyAllWindows()