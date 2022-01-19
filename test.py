import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

new_model = tf.keras.models.load_model('my_model')

a = input("Enter a champ: ")
a = a.split(',')
df1= pd.get_dummies(a)
df = np.asarray(df1).astype('float32')
data = tf.convert_to_tensor(df)

data2 = []
for i in data:
    
    data2.append(i.reshape((-1, 1)))


ab = new_model.predict(data[0].reshape((-1, 1)))

x = ab[0]
y = ab[1]
x2 = x[0] + x[1] /2
y2 = y[0] + y[1] /2

x3 = 0 - x2
y3 = 0 - y2

pp = np.array([x3, y3])
p2 = np.argmin(pp)
if p2 == 0:
    print(f"Prediction: {a[0]}")
if p2 == 1:
    print(f"Prediction: {a[1]}")
