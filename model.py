# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

print("si")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
cols = ["CHAMPION", "WIN"]
new_cols = ['vs_champ', 'win2']
data = pd.read_csv("../input/league-of-legends-1v1-matchups-results/matchups.csv", usecols=[3, 7])


d = dict()
for col, ncol in zip(cols, new_cols):
    d[col] = data[col].iloc[::2].values
    d[ncol] = data[col].iloc[1::2].values



df = pd.DataFrame(d)
target = df.pop("WIN")
target2 = df.pop("win2")
target = np.array(target, target2)
print(len(target))
target = np.asarray(target).astype('float32').reshape(-1, 1)


df1= pd.get_dummies(df)
df = np.asarray(df1).astype('float32')

data = tf.convert_to_tensor(df)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(data)
print(len(data))
normalizer(df1.iloc[:3])
data = data.reshape((-1,1))
data = data[:656126]
print(data)


def get_basic_model():
  model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['accuracy'])
  return model

model = get_basic_model()
model.fit(data, target, epochs=2, batch_size=BATCH_SIZE)
model.save('my_model')
