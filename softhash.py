### SoftHash !

import numpy as np

from keras.initializers import VarianceScaling
from keras.losses import mean_squared_error
from keras.optimizers import SGD

# recommendations for best tuning - not a must to use

softhash_initializer = VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=np.random.randint(0, 10**8))
softhash_loss = mean_squared_error
softhash_optimizer = SGD()

# ------------------------------------------------------------------------------------------- #

# 'subjectively' means that we're assuming neural networks already have abstract mathematical
# intelligence embedded in their structure. there's no better explanation of this giving
# better results than normal traning.

def fit_subjectively(model, train_x, train_y, _epochs=1, _batch_size=1, _verbose=0):
    perception = model.predict(train_x)
    
    model.fit(x=perception, y=train_y, epochs=_epochs, batch_size=_batch_size, verbose=_verbose)

def predict_subjectively(model, train_x):
    perception = model.predict(train_x)
    
    return model.predict(preception)

# ------------------------------------------------------------------------------------------- #

# converts the weights of a model/layer to machine readable signals of a chosen length

def softhash_weights(kObj, hashsize):
    obj_weights = kObj.get_weights()
    obj_weights = np.array(obj_weights)
    obj_weights = np.concatenate(obj_weights, axis=None)
    obj_weights = softhash_floats(inv_weights, hashsize, np.zeros(hashsize))

    return model_weights.reshape((1, hashsize))

# ------------------------------------------------------------------------------------------- #

# converts any raw string data to machine readable signals of a chosen length

def softhash(data, hashsize, allow_non_numeric=True):
    a = np.zeros(hashsize)
    chars = list(data)

    for x in range(0, len(chars)):
        cx = chars[x]
        sx = str(cx)
        if len(sx)==1:
            ascii_code = ord(cx)
            if allow_non_numeric or sx.isnumeric():
                wf = ascii_code if not sx.isnumeric() else float(sx) / 10 * 256
                if x >= hashsize:
                    a[x % hashsize] = (a[x % hashsize] + wf) / 2
                else:
                    a[x] = wf

    return a / 255

# ------------------------------------------------------------------------------------------- #

# reshapes machine readable signals to a chosen length
# arg 'a' is used to apply the hash over an existing hash of the same length
# may be initialized with np.zeros or other method

def softhash_floats(f, hashlen, a):
    for x in range(0, len(f)):
        a[x % hashlen] = (a[x % hashlen] + f[x]) / 2

    return a