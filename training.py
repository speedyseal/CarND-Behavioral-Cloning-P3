
# coding: utf-8

# In[ ]:

import importlib

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

import keras
from keras import backend as K
import train

import pandas as pd

keras.__version__


# In[ ]:

importlib.reload(train)


# In[ ]:

#csvpath = "./mydata/"
#image_path = "/mnt/g/gshare/carnd/fwd2/IMG/"
image_path = "./IMG/"
csvpath = "./fwd2/"
df=pd.read_csv(csvpath + "driving_log.csv", header=None, names=["center", "left", "right", "angle", "throttle", "brake", "speed"])

dfslow=pd.read_csv("./recoveryAndSlow.csv", header=None, names=["center", "left", "right", "angle", "throttle", "brake", "speed"])

df = pd.concat((df, dfslow))


# In[ ]:

### flip all images and their angles, concat to end
df['flip'] = False

dfflip = df.copy()
dfflip['angle'] = -dfflip['angle']
dfflip['flip'] = True
dftrain = pd.concat([df, dfflip])

print(df.iloc[0])
print(dfflip.iloc[0])


# In[ ]:

y_train = dftrain["angle"].values
n_train = len(y_train)
print("n_train:", n_train)


# In[ ]:

### Plot some statistics
print("mean:\t", np.mean(y_train))
print("median:\t", np.median(y_train))
print("std:\t", np.std(y_train))
print("mode:\t", scipy.stats.mode(y_train))

# y_trainall = np.hstack((y_train, -y_train))
plt.hist(y_train, bins=100)
plt.show()


# In[ ]:

y_train


# In[ ]:

pos_entry = df[df.angle > 0.].iloc[10]
neg_entry = df[df.angle < -0.1].iloc[10]; neg_entry
zero_entry = df[df.angle == 0.].iloc[10]; zero_entry

dfsub = pd.DataFrame([pos_entry, neg_entry, zero_entry])
dfsub = dfsub.append(pos_entry, ignore_index=True)
dfsub.at[3, 'flip'] = True

X_sub = train.loadImagesPd(dfsub, image_path)
plt.imshow(X_sub[0])
plt.show()

plt.imshow(X_sub[1])
plt.show()

dfsub


# In[ ]:

### check the cropping
model = train.nvidiaModel()
get_1st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output = get_1st_layer_output([X_sub])[0]
plt.imshow(layer_output[0]*0.5 - 0.5)
plt.show()


# In[ ]:

history=train.train_gen(model, dftrain, image_path, batch_size=128, epochs=20)
# help(model.fit_generator)


# In[ ]:

model.save('model.h5')

