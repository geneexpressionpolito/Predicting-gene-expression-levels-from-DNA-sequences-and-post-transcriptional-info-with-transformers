import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import stats
from keras.models import Model, load_model
import numpy as np
from matplotlib import pyplot
import datetime, os
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
import seaborn as sns
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

class TokenEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

    def call(self, x):
        x = self.token_emb(x)
        return x

class projCNN1D:
    def __init__(   self,
                    checkpoint_dir="",
                    model_type="Xpresso",
                    n_epochs=10, 
                    batch_size=32, 
                    learning_rate=5e-4,
                    momentum=0.9,
                    CNN_input=(10_500, 4),
                    miRNA_input=(2064,),
                    lr_reduction_epoch=None,
                    dropout_rate=0.4,
                    shuffle=True,
                    logdir=None,
                    patience=30,
                    opt = "SGD",
                    loss = "mse"):
        
        self.checkpoint_dir=checkpoint_dir
        self.model_type=model_type
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.CNN_input=CNN_input
        self.miRNA_input = miRNA_input
        self.dropout_rate = dropout_rate
        self.lr_reduction_epoch = lr_reduction_epoch
        self.shuffle=shuffle
        self.logdir=logdir
        self.patience=patience
        self.opt = opt
        self.loss = loss
        self.history = ""

        self._build_model()

    def _build_model(self):

        if self.model_type == "Xpresso_micro":
            #inputs
            input1 = layers.Input(shape=(self.CNN_input))
            input2 = layers.Input(shape=(8))
            input3 = layers.Input(shape=(self.miRNA_input))
            #CNN
            #1 layer
            x = layers.Conv1D(filters=128, kernel_size=6, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1)
            x = layers.MaxPooling1D(pool_size=30, strides=None, padding="valid")(x)
            #2 layer
            x = layers.Conv1D(filters=32, kernel_size=9, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.MaxPooling1D(pool_size=10, strides=None, padding="valid")(x)
       
            #Linear
            x = layers.Flatten()(x)

            x = layers.Concatenate()([x, input2, input3])
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.00099)(x)
            x = layers.Dense(2, activation="relu")(x)
            x = layers.Dropout(0.01546)(x)
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = keras.Model(
                inputs=[input1, input2, input3],
                outputs=[output],
                )
            self.model.summary()
            img = keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        if self.model_type == "Xpresso_TF":
            #inputs
            input1 = layers.Input(shape=(self.CNN_input))
            input2 = layers.Input(shape=(8))
            input3 = layers.Input(shape=(181))
            #CNN
            #1 layer
            x = layers.Conv1D(filters=128, kernel_size=6, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1)
            x = layers.MaxPooling1D(pool_size=30, strides=None, padding="valid")(x)
            #2 layer
            x = layers.Conv1D(filters=32, kernel_size=9, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.MaxPooling1D(pool_size=10, strides=None, padding="valid")(x)
       
            #Linear
            x = layers.Flatten()(x)
            x = layers.Concatenate()([x, input2, input3])
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.00099)(x)
            x = layers.Dense(2, activation="relu")(x)
            x = layers.Dropout(0.01546)(x)
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = keras.Model(
                inputs=[input1, input2, input3],
                outputs=[output],
                )
            self.model.summary()
            img = keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        if self.model_type == "Xpresso_ort":
            #inputs
            input1 = layers.Input(shape=(self.CNN_input))
            input2 = layers.Input(shape=(8))
            #CNN
            #1 layer
            initializer = tf.keras.initializers.Orthogonal( gain=1.0, seed=None )

            x = layers.Conv1D(filters=128, kernel_size=6, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer=initializer)(input1)
            x = layers.MaxPooling1D(pool_size=30, strides=None, padding="valid")(x)
            #2 layer
            x = layers.Conv1D(filters=32, kernel_size=9, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer=initializer)(x)
            x = layers.MaxPooling1D(pool_size=10, strides=None, padding="valid")(x)

            #Linear
            x = layers.Flatten()(x)
            x = layers.Concatenate()([x, input2])
            x = layers.Dense(64, activation="relu", kernel_initializer=initializer)(x)
            x = layers.Dropout(0.00099)(x)
            x = layers.Dense(2, activation="relu", kernel_initializer=initializer)(x)
            x = layers.Dropout(0.01546)(x)
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = keras.Model(
                inputs=[input1, input2],
                outputs=[output],
                )
            self.model.summary()
            img = keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        if self.model_type == "DivideEtImpera":
            #inputs
            input1  = layers.Input(shape=(self.CNN_input))
            input11 = layers.Input(shape=(8))

            ksize1 = 3
            ksize2 = 3
            ksize3 = 3

            pool1  = 5
            pool2  = 5
            pool3  = 5

            filter1 = 128
            filter2 = 128
            filter3 = 128
            filter4 = 128

            intervals = [a for a in range(0, self.CNN_input[0] + self.CNN_input[0]//10, self.CNN_input[0]//10)]
            
            emblayer = TokenEmbedding(10500, 4, 32)
            #embedding
            seq = input1
            #seq = emblayer(input1)

            #CNN
            #1 layer
            print(seq[:, 0:2000, :].shape)
            x1 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[0]:intervals[1], :])
            x1 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x1)
            x1 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x1)
            x1 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x1)
            x1 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x1)

            x2 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[1]:intervals[2], :])
            x2 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x2)
            x2 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x2)
            x2 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x2)
            x2 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x2)

            x3 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[2]:intervals[3], :])
            x3 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x3)
            x3 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x3)
            x3 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x3)
            x3 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x3)

            x4 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[3]:intervals[4], :])
            x4 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x4)
            x4 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x4)
            x4 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x4)
            x4 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x4)

            x5 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[4]:intervals[5], :])
            x5 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x5)
            x5 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x5)
            x5 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x5)
            x5 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x5)

            x6 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[5]:intervals[6], :])
            x6 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x6)
            x6 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x6)
            x6 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x6)
            x6 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x6)

            x7 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[6]:intervals[7], :])
            x7 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x7)
            x7 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x7)
            x7 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x7)
            x7 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x7)

            x8 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[7]:intervals[8], :])
            x8 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x8)
            x8 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x8)
            x8 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x8)
            x8 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x8)

            x9 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[8]:intervals[9], :])
            x9 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x9)
            x9 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x9)
            x9 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x9)
            x9 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x9)

            x10 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(seq[:, intervals[9]:intervals[10], :])
            x10 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x10)
            x10 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x10)
            x10 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x10)
            x10 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x10)

            x = layers.Concatenate(axis=-2)([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
            #2 layer
            x = layers.Conv1D(filters=filter3, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.MaxPooling1D(pool_size=pool3, strides=None, padding="valid")(x)
            x = layers.Conv1D(filters=filter4, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.Conv1D(filters=filter4, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.Conv1D(filters=filter4, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
     
            #Linear
            x = layers.Flatten()(x)
            x = layers.Concatenate()([x, input11])
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.00099)(x)
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.01546)(x)
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = keras.Model(
                inputs=[input1, input11],
                outputs=[output],
                )
            self.model.summary()
            img = keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        if self.model_type == "DivideEtImpera_TF":
            #inputs
            input1   = layers.Input(shape=(self.CNN_input))
            input11  = layers.Input(shape=(8))
            input12  = layers.Input(shape=(181))

            ksize1 = 3
            ksize2 = 3
            ksize3 = 3

            pool1  = 5
            pool2  = 5
            pool3  = 5

            filter1 = 128
            filter2 = 128
            filter3 = 128
            filter4 = 128

            intervals = [a for a in range(0, self.CNN_input[0] + self.CNN_input[0]//10, self.CNN_input[0]//10)]
            #CNN
            #1 layer
            print(input1[:, 0:2000, :].shape)
            x1 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[0]:intervals[1], :])
            x1 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x1)
            x1 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x1)
            x1 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x1)
            x1 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x1)

            x2 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[1]:intervals[2], :])
            x2 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x2)
            x2 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x2)
            x2 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x2)
            x2 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x2)

            x3 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[2]:intervals[3], :])
            x3 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x3)
            x3 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x3)
            x3 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x3)
            x3 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x3)

            x4 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[3]:intervals[4], :])
            x4 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x4)
            x4 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x4)
            x4 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x4)
            x4 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x4)

            x5 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[4]:intervals[5], :])
            x5 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x5)
            x5 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x5)
            x5 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x5)
            x5 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x5)

            x6 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[5]:intervals[6], :])
            x6 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x6)
            x6 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x6)
            x6 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x6)
            x6 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x6)

            x7 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[6]:intervals[7], :])
            x7 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x7)
            x7 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x7)
            x7 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x7)
            x7 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x7)

            x8 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[7]:intervals[8], :])
            x8 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x8)
            x8 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x8)
            x8 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x8)
            x8 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x8)

            x9 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[8]:intervals[9], :])
            x9 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x9)
            x9 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x9)
            x9 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x9)
            x9 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x9)

            x10 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[9]:intervals[10], :])
            x10 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x10)
            x10 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x10)
            x10 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x10)
            x10 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x10)

            x = layers.Concatenate(axis=-2)([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
            #2 layer
            x = layers.Conv1D(filters=filter3, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.MaxPooling1D(pool_size=pool3, strides=None, padding="valid")(x)
            x = layers.Conv1D(filters=filter4, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.Conv1D(filters=filter4, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.Conv1D(filters=filter4, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
     
            #Linear
            x = layers.Flatten()(x)
            x = layers.Concatenate()([x, input11, input12])
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.00099)(x)
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.01546)(x)
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = keras.Model(
                inputs=[input1, input11, input12],
                outputs=[output],
                )
            self.model.summary()
            img = keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        if self.model_type == "DivideEtImpera_onlyPromo":
            #inputs
            input1   = layers.Input(shape=(self.CNN_input))

            ksize1 = 3
            ksize2 = 3
            ksize3 = 3

            pool1  = 5
            pool2  = 5
            pool3  = 5

            filter1 = 128
            filter2 = 128
            filter3 = 128
            filter4 = 128

            intervals = [a for a in range(0, self.CNN_input[0] + self.CNN_input[0]//10, self.CNN_input[0]//10)]
            #CNN
            #1 layer
            print(input1[:, 0:2000, :].shape)
            x1 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[0]:intervals[1], :])
            x1 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x1)
            x1 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x1)
            x1 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x1)
            x1 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x1)

            x2 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[1]:intervals[2], :])
            x2 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x2)
            x2 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x2)
            x2 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x2)
            x2 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x2)

            x3 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[2]:intervals[3], :])
            x3 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x3)
            x3 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x3)
            x3 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x3)
            x3 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x3)

            x4 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[3]:intervals[4], :])
            x4 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x4)
            x4 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x4)
            x4 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x4)
            x4 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x4)

            x5 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[4]:intervals[5], :])
            x5 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x5)
            x5 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x5)
            x5 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x5)
            x5 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x5)

            x6 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[5]:intervals[6], :])
            x6 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x6)
            x6 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x6)
            x6 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x6)
            x6 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x6)

            x7 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[6]:intervals[7], :])
            x7 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x7)
            x7 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x7)
            x7 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x7)
            x7 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x7)

            x8 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[7]:intervals[8], :])
            x8 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x8)
            x8 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x8)
            x8 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x8)
            x8 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x8)

            x9 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[8]:intervals[9], :])
            x9 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x9)
            x9 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x9)
            x9 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x9)
            x9 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x9)

            x10 = layers.Conv1D(filters=filter1, kernel_size=ksize1, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1[:, intervals[9]:intervals[10], :])
            x10 = layers.MaxPooling1D(pool_size=pool1, strides=None, padding="valid")(x10)
            x10 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x10)
            x10 = layers.Conv1D(filters=filter2, kernel_size=ksize2, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x10)
            x10 = layers.MaxPooling1D(pool_size=pool2, strides=None, padding="valid")(x10)

            x = layers.Concatenate(axis=-2)([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
            #2 layer
            x = layers.Conv1D(filters=filter3, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.MaxPooling1D(pool_size=pool3, strides=None, padding="valid")(x)
            x = layers.Conv1D(filters=filter4, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.Conv1D(filters=filter4, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.Conv1D(filters=filter4, kernel_size=ksize3, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
     
            #Linear
            x = layers.Flatten()(x)
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.00099)(x)
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.01546)(x)
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = keras.Model(
                inputs=[input1],
                outputs=[output],
                )
            self.model.summary()
            img = keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        if self.model_type=="Xpresso_nohalf":
            #inputs
            input1 = layers.Input(shape=(self.CNN_input))
            #CNN
            #1 layer
            x = layers.Conv1D(filters=128, kernel_size=6, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1)
            x = layers.MaxPooling1D(pool_size=30, strides=None, padding="valid")(x)
            #2 layer
            x = layers.Conv1D(filters=32, kernel_size=9, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.MaxPooling1D(pool_size=10, strides=None, padding="valid")(x)
      
            #Linear
            x = layers.Flatten()(x)
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.00099)(x)
            x = layers.Dense(2, activation="relu")(x)
            x = layers.Dropout(0.01546)(x)
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = keras.Model(
                inputs=[input1],
                outputs=[output],
                )
            self.model.summary()
            img = keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)


        if self.model_type == "Xpresso":
            # params = { 'datadir' : datadir, 'batchsize' : 2**7, 'leftpos' : 3000, 'rightpos' : 13500, 'activationFxn' : 'relu', 'numFiltersConv1' : 2**7, 'filterLenConv1' : 6, 'dilRate1' : 1,
            # 'maxPool1' : 30, 'numconvlayers' : { 'numFiltersConv2' : 2**5, 'filterLenConv2' : 9, 'dilRate2' : 1, 'maxPool2' : 10, 'numconvlayers1' : { 'numconvlayers2' : 'two' } },
            # 'dense1' : 2**6, 'dropout1' : 0.00099, 'numdenselayers' : { 'layers' : 'two', 'dense2' : 2, 'dropout2' : 0.01546 } }
            #inputs
            input1 = layers.Input(shape=(self.CNN_input))
            input2 = layers.Input(shape=(8))
            #CNN
            #1 layer
            x = layers.Conv1D(filters=128, kernel_size=6, strides=1, padding="same", dilation_rate=1,  activation="relu", kernel_initializer='glorot_normal')(input1)
            x = layers.MaxPooling1D(pool_size=30, strides=None, padding="valid")(x)
            #2 layer
            x = layers.Conv1D(filters=32, kernel_size=9, strides=1, padding="same", dilation_rate=1, activation="relu", kernel_initializer='glorot_normal')(x)
            x = layers.MaxPooling1D(pool_size=10, strides=None, padding="valid")(x)

            #Linear
            x = layers.Flatten()(x)
            x = layers.Concatenate()([x, input2])
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(0.00099)(x)
            x = layers.Dense(2, activation="relu")(x)
            x = layers.Dropout(0.01546)(x)
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = keras.Model(
                inputs=[input1, input2],
                outputs=[output],
                )
            self.model.summary()
            img = keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        print(f"\nParameters:\n{vars(self)}\n")

    def train_model(self, x_train, y_train, x_val=None, y_val=None, TPU=False):
        #train test split
        if x_val is None:
            x_train, y_train, x_val, y_val = self._split_validation_data(x_train, y_train, 0.1)
        #optimizer
        if self.opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate, momentum=self.momentum)
        else:
            optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.loss)

        history = tf.keras.callbacks.History()
        check_cb = ModelCheckpoint(os.path.join(f"Saved_Models/checkpoint/{self.checkpoint_dir}", f'bestmodel_CNN1D_{self.model_type}'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        earlystop_cb = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1, mode='min', restore_best_weights=TPU)

        if TPU == True:
            if self.lr_reduction_epoch is not None:
                scheduler_callback = tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=1)
                callbacks = [history,
                                scheduler_callback,
                                earlystop_cb] 
            else:
                callbacks = [history,
                                earlystop_cb]
        else:
            if self.lr_reduction_epoch is not None:
                scheduler_callback = tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=1)
                callbacks = [history,
                                check_cb,
                                scheduler_callback,
                                earlystop_cb]
            else:
                callbacks = [history,
                                check_cb,
                                earlystop_cb]
        if self.logdir is not None and TPU is not True:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(self.logdir, 
                                                        histogram_freq=1,
                                                        write_grads=True,
                                                        update_freq='epoch')
            callbacks.append(tensorboard_callback)

        self.model.fit(x=x_train, 
            y=y_train, 
            shuffle=self.shuffle,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks)

        self.history = history
        plt.rcParams["figure.figsize"] = (20,9)
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.hlines(0.4, 0, len(history.history['loss']) , alpha = 0.2)
        pyplot.hlines(0.42, 0, len(history.history['loss']) , alpha = 0.2 )
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.show()

    def evaluate(self, x, y):
        predictions = self.model.predict(x).flatten()
        slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, y)
        print('Test R^2 = %.3f' % r_value**2)
        return r_value**2

    def evaluate_best(self, x, y, TPU=False):
        if TPU is False:
            best_file = os.path.join(f"Saved_Models/checkpoint/{self.checkpoint_dir}", f'bestmodel_CNN1D_{self.model_type}')
            model = load_model(best_file)
            predictions = model.predict(x).flatten()
        else:
            predictions = self.model.predict(x).flatten()
        slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, y)
        print('Test R^2 = %.3f' % r_value**2)
        return r_value**2

    def plot_kde(self, x, y, TPU=False):
        if TPU is False:
            best_file = os.path.join(f"Saved_Models/checkpoint/{self.checkpoint_dir}", f'bestmodel_CNN1D_{self.model_type}')
            model = load_model(best_file)
            predictions = model.predict(x).flatten()
        else:
            predictions = self.model.predict(x).flatten()
        df = pd.DataFrame({"predictions":predictions, "true":y})
        ax = sns.displot(data=df, kde=True)
        plt.xlabel("Labels")
        plt.show()
        
    def plot_train(self):
        history = self.history
        plt.rcParams["figure.figsize"] = (20,9)
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.hlines(0.4, 0, len(history.history['loss']) , alpha = 0.2)
        pyplot.hlines(0.42, 0, len(history.history['loss']) , alpha = 0.2 )
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.show()

    def plot_r2(self, x, y, TPU=False):
        from matplotlib import cm
        if TPU == False:
            best_file = os.path.join(f"Saved_Models/checkpoint/{self.checkpoint_dir}", f'bestmodel_CNN1D_{self.model_type}')
            model = load_model(best_file)
            predictions = model.predict(x).flatten()
        else:
            predictions = self.model.predict(x).flatten()
        slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, y)

        viridis = cm.get_cmap('autumn', 12)
        diff = y - predictions
        diff = np.abs(diff)

        ### plt size
        plt.rcParams["figure.figsize"] = (10,9)
        ### plt fontsize
        plt.rcParams.update({'font.size': 16})

        ### set title
        plt.title("Expression Scatterplot")
        ### plot
        bis = np.arange(-1.5, 3, 2)
        plt.plot(bis, bis,  f"b", alpha=0.3)
        for p, yi, c in zip(predictions, y, diff):
            plt.plot(p, yi,  f".", markersize=10, color=viridis((1.0-c)/1.1))
        ### set ticks
        plt.xticks([i for i in range(-1, 4)])
        plt.yticks([i for i in range(-1, 4)])
        ### set labels
        plt.xlabel("Predicted expression level")
        plt.ylabel("Median expression level")
        ### create legend
        plt.legend(loc="upper right", title=f"r2 = %.3f\n n = 1000" % r_value**2)
        ### set ylim
        plt.ylim((-1.5,3))
        plt.xlim((-1.5,3))
        ### grid
        plt.grid(alpha=0.5)
        ### save
        # if self.save:
        #     plt.savefig(f"{self.dir}{self.filename}.png")
        ### show
        plt.show()

    @staticmethod
    def _split_validation_data(x, y, validation_split):
        rand_indexes = np.random.permutation(x.shape[0])
        x = x[rand_indexes]
        y = y[rand_indexes]
        x_validation = x[:int(len(x) * validation_split)]
        y_validation = y[:int(len(x) * validation_split)]
        x_train = x[int(len(x) * validation_split):]
        y_train = y[int(len(x) * validation_split):]
        return x_train, y_train, x_validation, y_validation

    def lr_scheduler(self, epoch, lr):
        if epoch == self.lr_reduction_epoch:
            return lr * 0.1
        else:
            return lr 