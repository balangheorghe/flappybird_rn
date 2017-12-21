import tensorflow as tf
from keras import backend
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam


L1_constant = 0.01
L2_constant = 0.01
LEARNING_RATE = 1e-4
INPUT_SIZE = 8


def build_neural_network_model():
    model = Sequential()
    model.add(Dense(units=100, use_bias=True,
                    kernel_initializer="random_uniform", bias_initializer="zeros",
                    kernel_regularizer=regularizers.l2(L2_constant), bias_regularizer=regularizers.l1(L1_constant),
                    activity_regularizer=regularizers.l1_l2(L1_constant, L2_constant),
                    input_dim=INPUT_SIZE))
    model.add(Activation("relu"))
    model.add(Dense(units=2, use_bias=True,
                    kernel_initializer="random_uniform", bias_initializer="zeros",
                    kernel_regularizer=regularizers.l2(L2_constant), bias_regularizer=regularizers.l1(L1_constant),
                    activity_regularizer=regularizers.l1_l2(L1_constant, L2_constant)
                    ))
    model.add(Activation("relu"))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss="mse", optimizer=adam)
    return model

def main():
    model = build_neural_network_model()

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    backend.set_session(session=session)
    main()