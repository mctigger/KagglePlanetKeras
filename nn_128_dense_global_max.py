import os
import sys

from keras.models import Model
from keras.layers import Input, MaxPool2D, BatchNormalization, Dense, GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2

import fit_model
import load_data
import callbacks
from modules_dense import DenseBlock, CompressionBlock

name = os.path.basename(sys.argv[0])[:-3]

activation = 'elu'
bn_scale = False
dropout_rate = 0
l = 0.0001

inputs = pipe = Input(shape=(128, 128, 3))

pipe = DenseBlock(4, 8, dropout_rate, kernel_regularizer=l2(l))(pipe)
pipe = MaxPool2D((2, 2))(pipe)

pipe = CompressionBlock(32, kernel_regularizer=l2(l))(pipe)
pipe = DenseBlock(4, 16, dropout_rate, kernel_regularizer=l2(l))(pipe)
pipe = MaxPool2D((2, 2))(pipe)

pipe = CompressionBlock(64, kernel_regularizer=l2(l))(pipe)
pipe = DenseBlock(4, 32, dropout_rate, kernel_regularizer=l2(l))(pipe)
pipe = MaxPool2D((2, 2))(pipe)

pipe = CompressionBlock(128, kernel_regularizer=l2(l))(pipe)
pipe = DenseBlock(4, 64, dropout_rate, kernel_regularizer=l2(l))(pipe)
pipe = MaxPool2D((2, 2))(pipe)

pipe = CompressionBlock(256, kernel_regularizer=l2(l))(pipe)
pipe = DenseBlock(4, 128, dropout_rate, kernel_regularizer=l2(l))(pipe)
pipe = GlobalMaxPooling2D()(pipe)

pipe = BatchNormalization()(pipe)
pipe = Dense(17, activation='sigmoid', use_bias=False)(pipe)

model = Model(inputs=inputs, outputs=pipe)
model.compile(
    optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
    loss='binary_crossentropy',
    metrics=['binary_crossentropy', 'accuracy']
)

print(model.summary())


fit_model.fit_model(
    model,
    name,
    chain_train=[load_data.load(), load_data.augment(
        shear_fn=lambda: 0
    ), load_data.resize(128)],
    chain_val=[load_data.load(), load_data.resize(128)],
    epochs=200,
    batch_size=64,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001),
        callbacks.LossHistory(name, 'val_binary_crossentropy', save_model=True),
        callbacks.LossHistory(name, 'binary_crossentropy', save_model=False)
    ]
)