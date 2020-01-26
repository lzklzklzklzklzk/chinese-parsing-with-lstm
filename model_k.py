from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Dense, Flatten, LSTM, GRU, Bidirectional, Masking, TimeDistributed, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import preprocess
import config


if __name__ == '__main__':
    train_texts, train_labels, test_texts = preprocess.preprocess()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts + test_texts)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    word_index = tokenizer.index_word

    max_words = len(word_index)  # 词库总大小, 包含了train/test中所有字

    train_data = pad_sequences(
        train_sequences, maxlen=config.max_characters, value=0, padding='post')
    train_labels = pad_sequences(
        train_labels, maxlen=config.max_characters, value=0, padding='post')

    one_hot_train_labels = to_categorical(train_labels)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(config.max_characters,)))
    model.add(Embedding(input_dim=max_words,
                        output_dim=config.embedding_dim, mask_zero=True))
    model.add(Bidirectional(LSTM(config.LSTM_hidden_dims, return_sequences=True)))
    model.add(Bidirectional(LSTM(config.LSTM_hidden_dims, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(4, activation='softmax')))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['acc'])

    '''
    history = model.fit(train_data, one_hot_train_labels,
                        epochs=config.epochs, batch_size=config.batch_size, validation_split=0.2)

    model.save_weights('model_1.h5')
    '''
    model.load_weights('model_1.h5')

    test_sequences = tokenizer.texts_to_sequences(test_texts)

    test_data = pad_sequences(
        test_sequences, maxlen=config.max_characters, value=0, padding='post')

    test_pred = model.predict(test_data)

    print(test_pred)

    print(test_pred.shape)

    test_length = len(test_data)

    with open('../eval/test_results.txt', 'w') as f:
        for i in range(test_length):
            prediction = np.argmax(test_pred[i], axis=1)
            text = test_texts[i].replace(' ', '')
            l = len(text)
            line = ''
            for j in range(min(l, 300)):
                if prediction[j] == 0 or prediction[j] == 3:
                    line = line + '  ' + text[j]
                else:
                    line = line + text[j]
            if l >= 300:
                line = line + text[300:]
            line = line.strip()
            print(line)
            f.write(line + '\n')
