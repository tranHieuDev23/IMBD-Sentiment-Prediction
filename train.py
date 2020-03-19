import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from os import path


def train_model(X, y, model_filename, sequence_length, vocab_size, embedding_size, dropout_rate, lstm_output):
    X = pad_sequences(X, sequence_length)
    X = np.asarray(X)
    y = np.asarray(y)
    (dataset_size, sequence_length) = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = None
    if (path.exists(model_filename)):
        model = load_model(model_filename)
    else:
        model = Sequential([
            Embedding(vocab_size, embedding_size,
                      input_length=sequence_length),
            Dropout(dropout_rate),
            LSTM(lstm_output),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
    model.summary()
    batch_size = 32
    epochs = 10
    model.fit(X_train, y_train, batch_size, epochs)
    (loss, metrics) = model.evaluate(X_test, y_test)
    print('Loss: ' + str(loss) + ", accuracy: " + str(metrics))
    model.save(model_filename)
    return model
