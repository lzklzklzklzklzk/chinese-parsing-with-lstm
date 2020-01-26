import config
import numpy as np


def read_data(path):
    lines = []
    with open(path, encoding='utf-8', mode='r') as f:
        for line in f:
            lines.append(line)
    return lines


def labeling(sentences):
    # B = 0
    # E = 1
    # I = 2
    # S = 3
    length = len(sentences)
    lines = []
    labels = []

    for i in range(length):
        words = sentences[i].split('  ')
        sentence = ''
        label = []
        for word in words:
            l = len(word)
            if l == 1:
                sentence = sentence + ' ' + word
                label.append(3)
            else:
                for i in range(l):
                    sentence = sentence + ' ' + word[i]
                    if i == 0:
                        label.append(0)
                    elif i == l - 1:
                        label.append(1)
                    else:
                        label.append(2)
        lines.append(sentence)
        labels.append(np.array(label))
    return lines, labels


def process_test(sentences):
    length = len(sentences)
    lines = []

    for i in range(length):
        words = sentences[i].split('  ')
        sentence = ''
        for word in words:
            l = len(word)
            for i in range(l):
                sentence = sentence + ' ' + word[i]
        lines.append(sentence)
    return lines


def preprocess():
    train_texts = read_data(config.train_path)
    test_texts = read_data(config.test_path)
    train_lines, train_labels = labeling(train_texts)
    test_lines = process_test(test_texts)
    return train_lines, train_labels, test_lines
