import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU, ReLU
import random
import numpy as np
import matplotlib.pyplot as plt


def run_nn(training_set, test_set=None, num_epochs=5):
    keras.backend.clear_session()
    X_train, Y_train = training_set
    X_test, Y_test = test_set

    max_length = X_train.shape[2]
    num_features = X_train.shape[1]
    training_conv_nn = X_train.reshape(-1, 1, max_length, num_features)
    test_conv_nn = X_test.reshape(-1, 1, max_length, num_features)


    keras.backend.clear_session()

    model = Sequential()
    model.add(Conv2D(20, kernel_size=(1, 5), strides=(1, 1),
                     activation='linear',
                     input_shape=(1, max_length, num_features)))

    model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D(pool_size=(1, 5)))
    # model.add(ReLU())
    #     model.add(MaxPooling2D((1, 30),padding='same'))

    #     model.add(Conv2D(10, (1, 15), activation='linear'))
    #     model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(20, (1, 5), activation='linear'))

    model.add(ReLU())

    model.add(MaxPooling2D(pool_size=(1, 5)))

    # model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(270, activation='linear'))

    model.add(ReLU())

    #     model.add(Dropout(0.8))
    model.add(Dense(max_length, activation='sigmoid'))
    print(model.summary())
    #     return model
    opt = keras.optimizers.Adam(lr=0.01)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=opt,
                  metrics=['binary_accuracy', 'accuracy', 'categorical_accuracy'])

    history = model.fit(training_conv_nn, Y_train,
                        batch_size=32,
                        epochs=num_epochs,
                        validation_data=(test_conv_nn, Y_test))
    score = model.evaluate(test_conv_nn, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return model


