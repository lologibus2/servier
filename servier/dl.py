from keras.layers import BatchNormalization, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, concatenate
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from servier.data import EMBEDDING_VECTOR_LENGTH, MAX_LENGTH, VOCAB_SIZE


def mlp_model_1(size, verbose=False):
    model = Sequential([
        Dense(size, input_shape=(size,), activation="relu"),
        Dropout(0.2),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        BatchNormalization(axis=1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def mlp_model_2(size, verbose=False, *args, **kwargs):
    model = Sequential([
        Dense(512, input_shape=(size,), activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dropout(0.2),
        BatchNormalization(axis=1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def cnn_model(optimizer="adam", lr=0.001, dropout=0, layers=2,
              vocab_size=VOCAB_SIZE, embedding_vector_length=EMBEDDING_VECTOR_LENGTH, max_length=MAX_LENGTH, *args,
              **kwargs):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_vector_length, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    if layers == 3:
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    if optimizer == "adam" and lr != 0.001:
        print("Setting learning rate to" + str(lr))
        optimizer = Adam(lr)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model


def rnn_model(optimizer="adam", lr=0.001, dropout=0, gate="lstm", gated_layers=2, num_gated_connections=100,
              vocab_size=VOCAB_SIZE, embedding_vector_length=EMBEDDING_VECTOR_LENGTH, max_length=MAX_LENGTH, *args,
              **kwargs):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_vector_length, input_length=max_length))

    if gated_layers == 2:
        if gate == "lstm":
            model.add(LSTM(num_gated_connections, return_sequences=True, dropout=dropout))
        else:  # gate=gru
            model.add(GRU(num_gated_connections, return_sequences=True, dropout=dropout))

    if gate == "lstm":
        model.add(LSTM(num_gated_connections, dropout=dropout))
    else:  # gate=="gru"
        model.add(GRU(num_gated_connections, dropout=dropout))

    model.add(Dense(1, activation="sigmoid"))

    if optimizer == "adam" and lr != 0.001:
        print("Setting learning rate to" + str(lr))
        optimizer = Adam(lr)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model


def cnn_mlp_model(X1, X2, y, epochs=2, batch_size=2, optimizer="adam", lr=0.001, dropout=0.0, layers=2, fp="maccs",
                  vocab=VOCAB_SIZE, embedding_length=EMBEDDING_VECTOR_LENGTH, max_len=MAX_LENGTH):
    """
    Replaces the merged cnn_lstm_mlp and cnn_gru_mlp
    """
    cnn_model = Sequential()
    cnn_model.add(Embedding(vocab, embedding_length, input_length=max_len))
    cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    if layers == 3:
        cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(64))

    fc_model = Sequential()
    fc_model.add(Dense(256, activation="relu"))
    fc_model.add(Dropout(dropout))
    fc_model.add(Dense(64, activation="relu"))

    model_concat = concatenate([cnn_model.output, fc_model.output], axis=-1)
    model_concat = Dense(64)(model_concat)
    model = Model(inputs=[model1.input, model2.input], outputs=model_concat)

    # concatenated = concatenate([cnn_model, fc_model])
    # model = Dense(64)(concatenated)
    # model = Sequential()
    # model.add(Merge([cnn_model, fc_model], mode='concat'))
    # model.add(Dense(64))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))

    if optimizer == "adam" and lr != 0.001:
        print("Setting learning rate to" + str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    model.fit([X1, X2], y, validation_split=0.1, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    return model
