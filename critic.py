import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

batch_size = 24
max_tokens = 20000

text_vectorization = layers.TextVectorization(
    output_mode = "multi_hot",
    ngrams = 2,
    max_tokens = max_tokens
)

def display(history):
    # history = history[10:]

    acc = history.history["mae"]
    val_acc = history.history["val_mae"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(2, len(acc) + 1)

    acc.pop(0)
    val_acc.pop(0)
    loss.pop(0)
    val_loss.pop(0)

    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

# Make numpy values easier to read.
#Look at later
np.set_printoptions(precision=3, suppress=True)

dataset = pd.read_csv("metacriticSet.csv")

#Remove unused info
dataset.pop("platform")
dataset.pop("user_review")
dataset.pop("summary")
dataset.pop("release_date")

#Check if they are shuffled correctly if starts fucking up
dataset = dataset.sample(
    frac=1
    ).reset_index()

#Remove "index" field. It wasn't there before for some reason
print(dataset.head())

scores_output = np.array(dataset.pop("meta_score"))
names_unclean = np.array(dataset)

names_input = np.array([])

#Remove the weird row number. Probably a better way somewhere
for i in names_unclean:
    names_input = np.append(names_input, i[1])

text_vectorization.adapt(names_input)

x_train, x_test, y_train, y_test = train_test_split(names_input, scores_output, test_size=0.2, shuffle=False)

# train_dataset = tf.data.Dataset.from_tensor_slices((names_input, scores_output))

#Create model here
inputs = keras.Input(shape=(1, ), dtype=tf.string)
x = text_vectorization(inputs)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(
    loss='mse',
    optimizer = 'rmsprop',
    metrics = ['mae']
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = r"checkpoints/DenseBigrams.tf",
        save_best_only = True,
        monitor = "val_mae"
    ),
]

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

history = model.fit(
    x_train,
    y_train,
    validation_data = (x_test, y_test),
    epochs = 25,
    callbacks = callbacks
)

print(model.predict(('Fighter Frogs', )))

display(history)