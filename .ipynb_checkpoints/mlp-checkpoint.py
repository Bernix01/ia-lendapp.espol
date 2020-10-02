import os
os.environ["KERAS_BACKEND"]="plaidml.keras.backend"
os.environ["PLAIDML_NATIVE_PATH"]="/usr/local/lib/libplaidml.dylib"
os.environ["RUNFILES_DIR"]="/usr/local/share/plaidml"
os.environ["PLAIDML_USE_STRIPE"]="1"
import pydot
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import numpy as np
from keras.utils import to_categorical
from keras.layers.preprocessing import CategoryEncoding
from keras.layers.preprocessing import StringLookup
from keras.layers.preprocessing import Normalization

# Select categorical features
cat_features = ["term", "is_risky","purpose","emp_title","disbursement_method","application_type"]





def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_string_categorical_feature(feature, name, dataset):
    # Create a StringLookup layer which will turn strings into integer indices
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):
    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(feature)
    return encoded_feature








model = Sequential()
model.add(Dense(50, activation='relu', input_dim=X_train_enc.shape[1]))
#model.add(Dropout(0.25))
model.add(Dense(30, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss = 'categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
model.summary()

batch_size = 1000
epochs = 10



model.fit(X_train_enc, y_train_enc,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=test)

score = model.evaluate(X_test_enc,
                       y_test_enc,
                       verbose=2)

print('Test loss:', score[0])
print('Test accuracy:', score[1])