# ------------------------------------------------------------------------
# Copyright (c) 2024 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
CLINet model architecture
"""
from tensorflow import keras
from involution import Involution
from tensorflow.keras.optimizers import Adam

def create_lstm_inv_model(num_classes = 8):
    inputs = keras.layers.Input(shape=(1250,1))

    x11 = keras.layers.Conv1D(filters=3, kernel_size=31,strides = 5,padding='same', activation=None, use_bias=True)(inputs)
    x12 = keras.layers.Conv1D(filters=3, kernel_size=36,strides = 5,padding='same', activation=None, use_bias=True)(inputs)
    x13 = keras.layers.Conv1D(filters=3, kernel_size=41,strides = 5,padding='same', activation=None, use_bias=True)(inputs)

    conv = keras.layers.concatenate([x11, x12, x13], axis = -1)

    x = keras.layers.BatchNormalization()(conv)
    x = keras.layers.ReLU()(x)
    x = keras.layers.TimeDistributed(Flatten())(x)
    x = keras.layers.LSTM(200, return_sequences=True)(x)

    x = keras.layers.Reshape((250, 200, 1))(x)

    x21, _ = Involution(channel=3, group_number=1, kernel_size=31, stride=5, reduction_ratio=2, name = '1')(x)
    x22, _ = Involution(channel=3, group_number=1, kernel_size=36, stride=5, reduction_ratio=2, name = '2')(x)
    x23, _ = Involution(channel=3, group_number=1, kernel_size=41, stride=5, reduction_ratio=2, name = '3')(x)

    inv = keras.layers.concatenate([x21, x22, x23], axis = -1)

    x = keras.layers.BatchNormalization()(inv)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(20)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.ReLU()(x)
    outputs = keras.layers.Dense(num_classes)(x)

    conv_lstm_inv_model = keras.Model(inputs=[inputs], outputs=[outputs], name="conv_lstm_inv_model")
    conv_lstm_inv_model.compile(optimizer=Adam(learning_rate=0.001),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return conv_lstm_inv_model
