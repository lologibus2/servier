from keras.layers import BatchNormalization, Dropout
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def get_model(size, verbose=False):
    print(size)
    model = Sequential([
        Dense(size, input_shape=(size,), activation="relu"),
        Dropout(0.2),
        Dense(512, activation="sigmoid"),
        Dropout(0.2),
        Dense(128, activation="sigmoid"),
        Dropout(0.2),
        Dense(16, activation="sigmoid"),
        BatchNormalization(axis=1),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    return model