from tensorflow import keras

model = keras.models.load_model('checkpoints/DenseBigrams.tf')

question = input('Make up a video game title\n')

print(model.predict((question, )))
