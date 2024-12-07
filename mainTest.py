import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('D:\Brain_tumor-detection\BrainTumor10Epochs.h5')

image = cv2.imread('D:\Brain_tumor-detection\pred\pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

# print(img)
result = (model.predict(input_img) > 0.5).astype("int32")
print(result)