import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from PIL import Image
import seaborn as sns
from sklearn.svm import SVR
import xgboost as xgb

TARGET = 'Pawpularity'
VAL_SIZE = 0.2

# TensorFlow settings and training parameters
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 224
BATCH_SIZE = 64
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.1
DECAY_STEPS = 100
DECAY_RATE = 0.95
EPOCHS = 1
PATIENCE = 5

base_model = '../input/keras-pretrained-models/ResNet152_Top_ImageNet.h5'
base_model = tf.keras.models.load_model(base_model)
# Freeze weights in the original model
base_model.trainable = False


train_csv = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
test_csv = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')
ids = train_csv.pop('Id')
corr = np.corrcoef(train_csv,rowvar=False)
ax = sns.heatmap(corr)
train_csv['Id'] = ids


# Image data directories
TRAIN_DIRECTORY = '../input/petfinder-pawpularity-score/train'
TEST_DIRECTORY = '../input/petfinder-pawpularity-score/test'

# Reconstruct the paths to train and test images.
train_csv['path'] = train_csv['Id'].apply(lambda x: os.path.join(TRAIN_DIRECTORY, f'{x}.jpg'))
train_csv[TARGET] = train_csv[TARGET].astype('float32')
test_csv['path'] = test_csv['Id'].apply(lambda x: os.path.join(TEST_DIRECTORY, f'{x}.jpg'))

rand_pic = np.random.randint(9912, size=(12))
fig, ax = plt.subplots(nrows=3, ncols=4,figsize = (14,8))

for i, pic in enumerate(rand_pic):
    r, c = i//4, i%4
    ax[r,c].imshow(np.asarray(Image.open(train_csv['path'][pic])))
    ax[r,c].axis('off')
    label = train_csv['Pawpularity'][pic]
    ax[r,c].set_title(f'Pawpularity Score: {label}', fontsize = 12, fontfamily='monospace', fontweight='bold')

fig, ax = plt.subplots(nrows=2, ncols=4,figsize = (14,8))
for pic in range(8):
    r, c = pic//4, pic%4
    ax[r,c].imshow(np.asarray(Image.open(test_csv['path'][pic])))
    ax[r,c].axis('off')
    label = '?'
    ax[r,c].set_title(f'Pawpularity Score: {label}', fontsize = 12, fontfamily='monospace', fontweight='bold')


@tf.function
def get_image(path: str) -> tf.Tensor:
    """Function loads image from a file and preprocesses it.
    :param path: Path to image file
    :return: Tensor with preprocessed image
    """
    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.cast(tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE), dtype=tf.float32)
    return tf.keras.applications.resnet.preprocess_input(image)

@tf.function
def get_image_wscore(path: str, score: int) -> tuple:
    """Function loads image from a file and preprocesses it.
    :param path: Path to image file
    :param score: Pawpularity Score
    :return: Tuple of Tensor with preprocessed image and Pawpularity score
    """
    return get_image(path), score

@tf.function
def get_dataset(x, y=None) -> tf.data.Dataset:
    """Function returns preprocessed image and label.
    :param x: Path to image file
    :param y: Pawpularity score if known
    :return: tf.Tensor with preprocessed image and numeric score.
    """
    if y is not None:
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        return ds.map(get_image_wscore, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(x)
        return ds.map(get_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)



# Base model: Resnet152
image_model_base = tf.keras.models.Sequential(
    [tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
    base_model
    ])

# Learning Rate Design - Exponential Decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
   initial_learning_rate=LEARNING_RATE,
   decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE,
   staircase=True)

# Compile the model
image_model_base.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])
image_model_base.summary()

# Early Stopping Rule
# It shows that 5 Epochs is enough.
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

# Split train dataset into a training subset(80%) and a validation subset(20%).
train_subset, valid_subset = train_test_split(
    train_csv[['path', TARGET]],
    test_size=VAL_SIZE, shuffle=True)

# Create TensorFlow datasets
train_ds = get_dataset(x=train_subset['path'], y=train_subset[TARGET])
valid_ds = get_dataset(x=valid_subset['path'], y=valid_subset[TARGET])
test_ds = get_dataset(x=test_csv['path'])

# Prediction based on Base model
print('Start predictions - Base Model - train')
base_pred_train = image_model_base.predict(train_ds)
base_pred_valid = image_model_base.predict(valid_ds)
base_pred_test = image_model_base.predict(test_ds)
print('Prediction Completed')



# XGBoost Tuning
# min_rmse = float("Inf")
# best_params = None
# dtrain = xgb.DMatrix(base_pred_train, label=train_subset[TARGET])

params = {'max_depth':5,
          'min_child_weight': 6,
          'alpha':6,
          'colsample_bytree':0.5,
          'eta':0.1
         }

# gridsearch_params = [(max_depth, min_child_weight,alpha)
#                      for max_depth in range(4,7)
#                      for min_child_weight in range(4,7)
#                      for alpha in range(4,7)
#                     ]

# Find best params for XGBoost
# for max_depth, min_child_weight, alpha in gridsearch_params:
#     print("CV with max_depth={}, min_child_weight={}, alpha={}".format(
#         max_depth,min_child_weight,alpha))
#     # Update our parameters
#     params['max_depth'] = max_depth
#     params['min_child_weight'] = min_child_weight
#     params['alpha'] = alpha
#     # Run CV
#     cv_results = xgb.cv(params,dtrain,seed = 40,nfold = 5,metrics = {'rmse'}, early_stopping_rounds = 5)
#     # Update best RMSE
#     mean_rmse = cv_results['test-rmse-mean'].min()
#     boost_rounds = cv_results['test-rmse-mean'].argmin()
#     print("\tRMSE {} for {}th rounds".format(mean_rmse, boost_rounds))
#     if mean_rmse < min_rmse:
#         min_rmse = mean_rmse
#         best_params = (max_depth,min_child_weight,alpha)

# print("Best params: {}, {}, {}, RMSE: {}".format(best_params[0], best_params[1], best_params[2], min_rmse))

# xgb_model = xgb.XGBRegressor(max_depth = best_params[0], min_child_weight = best_params[1], alpha = best_params[2],
#                              colsample_bytree = params['colsample_bytree'], eta = params['eta'], n_estimators = 100)

# Put the best params in the final model
xgb_model = xgb.XGBRegressor(max_depth = params['max_depth'], min_child_weight = params['min_child_weight'],
                             alpha = params['alpha'], colsample_bytree = params['colsample_bytree'],
                             eta = params['eta'], n_estimators = 100)
xgb_model.fit(base_pred_train, train_subset[TARGET])


# SVR Tuning
param_grid = {'C': [20], 'gamma': [0.8], 'kernel': ['poly','rbf','sigmoid']}
svr_model = GridSearchCV(SVR(),param_grid, refit=True, verbose=2)
# Put the best params in the final model
svr_model.fit(base_pred_train, train_subset[TARGET])


# Fit Neural Network
nn_model = image_model_base
nn_model.add(tf.keras.layers.BatchNormalization())
nn_model.add(tf.keras.layers.Dropout(DROPOUT_RATE, name='top_dropout'))
nn_model.add(tf.keras.layers.Dense(32, activation='relu'))
nn_model.add(tf.keras.layers.Dropout(DROPOUT_RATE, name='top_dropout2'))
nn_model.add(tf.keras.layers.Dense(1, name='score'))

# Compile the Neural Network model
# Use MSE as Loss Function.
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])
nn_model.summary()

history = nn_model.fit(train_ds, validation_data = valid_ds,
                          epochs=EPOCHS,# callbacks=[early_stop],
                          use_multiprocessing=True, workers=-1)


# Making Predictions based on new models for the validation dataset.
print('Start predictions - New Models - Valid')
nn_valid = nn_model.predict(valid_ds, use_multiprocessing=True, workers=os.cpu_count())
svr_valid = svr_model.predict(base_pred_valid)
xgb_valid = xgb_model.predict(base_pred_valid)
three_pred_valid = [[x[0] for x in nn_valid],svr_valid,xgb_valid]
print('Prediction Completed')


# Performance on validation dataset - RMSE
rmses = list(map(lambda x: np.sqrt( np.mean( (valid_subset[TARGET] - np.array(x))**2.0 ) ),three_pred_valid))
print('NN RSME =',rmses[0],'\n')
print('SVR RSME =',rmses[1],'\n')
print('XGB RSME =',rmses[2],'\n')


# Find best sets of weights with lowest rmse.
best_weights = None
lowest_rsme = float('Inf')
weights = [(nn,svr)
           for nn in range(1,8)
           for svr in range(1,8)]
for nn, svr in weights:
    if nn + svr <=10:
        xgb = 10-nn-svr
        pred = np.dot([nn,svr,xgb],[three_pred_valid])/10
        rmse = np.sqrt(np.mean((valid_subset[TARGET]-pred.reshape(valid_subset[TARGET].shape))**2))
        if rmse<lowest_rsme:
            lowest_rsme = rmse
            best_weights = [nn,svr,xgb]

print(lowest_rsme)
print(best_weights)


# Making Predictions for the test dataset.
print('Start predictions - New Models - Test')
test_csv['nn_Pawpularity'] = nn_model.predict(test_ds, use_multiprocessing=True, workers=os.cpu_count())
test_csv['svr_Pawpularity'] = svr_model.predict(base_pred_test)
test_csv['xgb_Pawpularity'] = xgb_model.predict(base_pred_test)
print('Prediction Completed')

# Generate Final Predictions and Save it.
test_csv[TARGET] = np.dot(test_csv[['nn_Pawpularity','svr_Pawpularity','xgb_Pawpularity']],best_weights)/10
test_csv[['Id', TARGET]].to_csv('submission.csv', index=False)
test_csv
