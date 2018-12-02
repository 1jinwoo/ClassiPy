from keras.models import Sequential
from keras import layers
import numpy as np
## SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


# plot history function
def plot_history(history, model=None, train_accuracy=None, test_accuracy=None, neurons=None, dropout_percentage=None, epoch=None):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    if model is not None:
        plt.title(model+', train accuracy = '+str(train_accuracy)+', test_accuracy = '+str(test_accuracy))
    else:
        plt.title('Training and validation loss')
    plt.legend()
    if model is not None:
        plt.savefig('images/'+model+'Neurons'+str(neurons)+' Dropout'+str(dropout_percentage)+' Epoch'+str(epoch)+'.jpg')
    else:
        plt.show()
    plt.close()


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix