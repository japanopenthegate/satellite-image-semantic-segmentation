import tensorflow as tf
from keras.models import load_model
import segmentation_models as sm

sm.set_framework('tf.keras')

sm.framework()
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras

from keras.utils.np_utils import normalize
from keras.metrics import MeanIoU

# Resizing images, if needed
SIZE_X = 128
SIZE_Y = 128
n_classes = 2  # Number of classes for segmentation

# Capture training image info as a list
train_images = []

for directory_path in glob.glob("data/train_img/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 1)
        # img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)

# Convert list to array for machine learning processing
X_train = np.array(train_images)

# Capture mask/label info as a list
train_masks = []
for directory_path in glob.glob("data/train_label/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)
        # mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)

# Convert list to array for machine learning processing
train_masks = np.array(train_masks) / 255

###############################################
# Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

y_train = np.expand_dims(train_masks_encoded_original_shape, axis=3)

from tensorflow.keras.utils import to_categorical

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

######################################################
# Reused parameters in all models

n_classes = 2
activation = 'softmax'

LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

########################################################################
BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# preprocess input
X_train2 = preprocess_input2(X_train)


# define model
model2 = sm.Unet(BACKBONE2, encoder_weights=load_model('inceptionv3_backbone_10epochs.hdf5',compile=True), classes=n_classes, activation=activation)


# compile keras model with defined optimozer, loss and metrics
model2.compile(optim, total_loss, metrics)
#model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)


print(model2.summary())


history2=model2.fit(X_train2,
          y_train_cat,
          batch_size=8,
          epochs=10,
          verbose=1,
          )


model2.save('inceptionv3_backbone_1epochs.hdf5')


from keras.models import load_model
import tensorflow as tf
import segmentation_models as sm

sm.set_framework('tf.keras')

sm.framework()
import glob
import cv2
import os
import numpy as np

### FOR NOW LET US FOCUS ON A SINGLE MODEL

# Set compile=False as we are not loading it for training, only for prediction.

BACKBONE3 = 'inceptionv3'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

model3 = load_model('inceptionv3_backbone_10epochs.hdf5',compile=False)

test_images = []

for directory_path in glob.glob("data/test_img/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 1)
        # img = cv2.resize(img, (SIZE_Y, SIZE_X))
        test_images.append(img)

X_test = np.array(test_images)
X_test2 = preprocess_input3(X_test)

y_pred1 = model3.predict(X_test2)
y_pred1_argmax = np.argmax(y_pred1, axis=3)

y_pred1_argmax=y_pred1_argmax.astype(np.uint8)
y_pred1_argmax=y_pred1_argmax*255

test_path = next(os.walk('data/test_img/'))[2]

for x,i in enumerate(test_path):

    cv2.imwrite('inception8/'+i,y_pred1_argmax[x])




model4 = load_model('inceptionv3_backbone_1epochs_2ndrun.hdf5',compile=False)

n_classes = 2
activation = 'softmax'

LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)
model4.compile(optim, total_loss, metrics)
#model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)


print(model4.summary())

history3=model4.fit(X_train2,
          y_train_cat,
          batch_size=8,
          epochs=12,
          verbose=1,
          )


model4.save('inceptionv3_backbone_1epochs_3rddrun.hdf5')
