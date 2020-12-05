from keras.layers import BatchNormalization
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def get_model(size, verbose=False):
    print(size)
    model = Sequential([
        Dense(size, input_shape=(size,), activation="relu"),
        Dense(256, activation="sigmoid"),
        Dense(64, activation="sigmoid"),
        Dense(34, activation="sigmoid"),
        Dense(16, activation="sigmoid"),
        BatchNormalization(axis=1),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer=Adam(lr=0.00001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model