# library import
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers

# file import
import data_cleaner as dc
import model_helper as mh

df = dc.clean_item_data(0)
df = dc.cleanup_categoryid(df)[0]
_X_train, _, _X_test, Y_train, _, Y_test = dc.data_split(df, 0.8, 0, 0.2)


# convert pandas dataframes to list of strings
x_train_list = []
x_test_list = []
for _, row in _X_train.iterrows():
    x_train_list.append(row[0])
for _, row in _X_test.iterrows():
    x_test_list.append(row[0])

# tokenize training input data based on pre-trained word embeddings
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train_list)
X_train = tokenizer.texts_to_sequences(x_train_list)
X_test = tokenizer.texts_to_sequences(x_test_list)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

maxlen = 15
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embedding_dim = 50
embedding_matrix = mh.create_embedding_matrix(
    'glove.6B/glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)
output_dim = df['categoryId'].nunique()
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.LSTM(units=50))
# model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dense(output_dim, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    epochs=8,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    batch_size=50)
model.summary()

loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
mh.plot_history(history)



