import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from scipy import stats
from matplotlib import pyplot
import numpy as np
import datetime, os
# %pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import math
import h5py
import pickle
import seaborn as sns
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
except Exception:
  pass
# Load the TensorBoard notebook extension
# %load_ext tensorboard
# Clear any logs from previous runs
!rm -rf ./logs/

"""# keras """

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=8000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def fft2d(x):   
    return tf.math.real(tf.signal.fft2d(tf.cast(x, complex64)))

class FNETBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, rate=0.1, compression=False):
        super(FNETBlock, self).__init__()
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.compression = compression

    def call(self, inputs, training):
        fft2D = fft2d(inputs)
        fft2D = self.dropout1(fft2D, training=training)
        out1 = self.layernorm1(inputs + fft2D)

        if self.compression is not False:
            truncated = tf.cast((out1.shape[1]*self.compression), tf.int32)
            out1 = out1[:, :truncated, :]

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TokenAndPositionEmbedding2(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding2, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions, tf.expand_dims(positions, axis=0)

class TokenAndPositionEmbedding3(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, rate=0.1):
        super(TokenAndPositionEmbedding3, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x) + positions
        x = self.dropout(x, training=training)
        return x

class TokenAndPositionEmbedding4(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, rate=0.1):
        super(TokenAndPositionEmbedding4, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_encoding = positional_encoding(maxlen,
                                            embed_dim)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.d_model = embed_dim

    def call(self, x, training):
        maxlen = tf.shape(x)[-1]
        x = self.token_emb(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :maxlen, :]
        x = self.dropout(x, training=training)
        return x

#only new_embedding
class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

#only new_embedding
class PositionEncoding2(layers.Layer):
    def __init__(self, maxlen, embed_dim, rate):
        super(PositionEncoding2, self).__init__()
        self.pos_encoding = positional_encoding(maxlen,
                                            embed_dim)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.d_model = embed_dim

    def call(self, x, training):
        maxlen = tf.shape(x)[-2]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :maxlen, :]
        x = self.dropout(x, training=training)
        return x

class projTransformer:
    def __init__(   self,
                    checkpoint_dir="",
                    model_type="best",
                    n_epochs=300, 
                    batch_size=32, 
                    learning_rate=1e-4,
                    momentum=0.9,
                    maxlen=10500,
                    embed_dim=32,
                    num_heads=4,
                    ff_dim=64,
                    vocab_size=5,
                    dense=64,
                    lr_reduction_epoch=None,
                    dropout_rate=0.1,
                    t_rate = 0.1,
                    patience=20,
                    optimizer="SGD",
                    warmup_steps = 8000,
                    shuffle = True,
                    loss = "mse",
                    logdir=None):
        
        self.checkpoint_dir     = checkpoint_dir
        self.model_type         = model_type
        self.n_epochs           = n_epochs
        self.batch_size         = batch_size
        self.learning_rate      = learning_rate
        self.momentum           = momentum
        self.maxlen             = maxlen
        self.embed_dim          = embed_dim
        self.num_heads          = num_heads
        self.ff_dim             = ff_dim
        self.vocab_size         = vocab_size
        self.dense              = dense
        self.dropout_rate       = dropout_rate
        self.lr_reduction_epoch = lr_reduction_epoch
        self.t_rate             = t_rate
        self.patience           = patience
        self.optimizer          = optimizer
        self.warmup_steps       = warmup_steps
        self.shuffle            = shuffle
        self.logdir             = logdir
        self.loss               = loss
        self.history            = ""

        self._build_model()

        #optimizer
        if self.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        if self.optimizer == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate, rho=0.95, epsilon=1e-07, name="Adadelta")
        if self.optimizer == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")
        if self.optimizer == "Original":
            learning_rate = CustomSchedule(self.embed_dim, self.warmup_steps)
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # compile    
        self.model.compile(optimizer=optimizer, loss=self.loss)

    def _build_model(self):
       

        if self.model_type == "DeepLncLoc":
            embedding_layer = PositionEncoding2(self.maxlen, self.embed_dim, self.t_rate)
            # embedding_layer = PositionEmbedding(self.maxlen, self.vocab_size,self.embed_dim)
            
            #inputs
            input1 = layers.Input(shape=(self.maxlen, self.embed_dim))
            input2 = layers.Input(shape=(8))

            x = input1
            #embedding
            x = layers.BatchNormalization()(x)
            x = embedding_layer(x)

            #transformers
            x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.t_rate)(x)

            #FC
            x = layers.GlobalAveragePooling1D()(x)
            # x = layers.Flatten()(x)

            x = layers.Concatenate()([x, input2])
            #dense1
            x = layers.Dense(self.dense, activation="relu")(x)
            x = layers.Dropout(self.dropout_rate)(x)

            #output
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = tf.keras.Model(
                inputs=[input1, input2],
                outputs=[output],
                )
            self.model.summary()
            img = tf.keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        if self.model_type == "DeepLncLoc_TF":
            embedding_layer = PositionEncoding2(self.maxlen, self.embed_dim, self.t_rate)
            # embedding_layer = PositionEmbedding(self.maxlen, self.vocab_size,self.embed_dim)
            
            #inputs
            input1 = layers.Input(shape=(self.maxlen, self.embed_dim))
            input2 = layers.Input(shape=(8))
            input3 = layers.Input(shape=(181))

            x = input1
            #embedding
            x = layers.BatchNormalization()(x)
            x = embedding_layer(x)

            #transformers
            x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.t_rate)(x)

            #FC
            x = layers.GlobalAveragePooling1D()(x)
            # x = layers.Flatten()(x)

            x = layers.Concatenate()([x, input2, input3])
            #dense1
            x = layers.Dense(self.dense, activation="relu")(x)
            x = layers.Dropout(self.dropout_rate)(x)

            #output
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = tf.keras.Model(
                inputs=[input1, input2, input3],
                outputs=[output],
                )
            self.model.summary()
            img = tf.keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        if self.model_type == "DeepLncLoc_onlyPromo":
            embedding_layer = PositionEncoding2(self.maxlen, self.embed_dim, self.t_rate)
            # embedding_layer = PositionEmbedding(self.maxlen, self.vocab_size,self.embed_dim)
            
            #inputs
            input1 = layers.Input(shape=(self.maxlen, self.embed_dim))

            x = input1
            #embedding
            x = layers.BatchNormalization()(x)
            x = embedding_layer(x)

            #transformers
            x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.t_rate)(x)

            #FC
            x = layers.GlobalAveragePooling1D()(x)
            # x = layers.Flatten()(x)

            #dense1
            x = layers.Dense(self.dense, activation="relu")(x)
            x = layers.Dropout(self.dropout_rate)(x)

            #output
            output = layers.Dense(1, activation="linear")(x)

            print("model built")
            self.model = tf.keras.Model(
                inputs=[input1],
                outputs=[output],
                )
            self.model.summary()
            img = tf.keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        

        

            print("model built")
            self.model = tf.keras.Model(
                inputs=[input1],
                outputs=[output],
                )
            self.model.summary()
            img = tf.keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
            display(img)

        print(f"\nParameters:\n{vars(self)}\n")

    def train_model(self, x_train, y_train, x_val=None, y_val=None, TPU=False):
        #train test split
        if x_val is None:
            x_train, y_train, x_val, y_val = self._split_validation_data(x_train, y_train, 0.1)

        history = tf.keras.callbacks.History()
        check_cb = ModelCheckpoint(os.path.join(f"Saved_Models/checkpoint/{self.checkpoint_dir}", f'bestmodel_transformer_{self.model_type}'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        earlystop_cb = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1, mode='min', restore_best_weights=TPU)

        if TPU == True:
            if self.lr_reduction_epoch is not None and self.optimizer is not "Original":
                scheduler_callback = tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=1)
                callbacks = [history,
                                scheduler_callback,
                                earlystop_cb] 
            else:
                callbacks = [history,
                                earlystop_cb]
        else:
            if self.lr_reduction_epoch is not None and self.optimizer is not "Original":
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

    def evaluate_best(self, x, y, TPU=False):
        if TPU is False:
            best_file = os.path.join(f"Saved_Models/checkpoint/{self.checkpoint_dir}", f'bestmodel_transformer_{self.model_type}')
            model = load_model(best_file)
            predictions = model.predict(x).flatten()
        else:
            predictions = self.model.predict(x).flatten()
        slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, y)
        print('Test R^2 = %.3f' % r_value**2)
        return r_value**2

    def plot_kde(self, x, y, TPU=False):
        if TPU is False:
            best_file = os.path.join(f"Saved_Models/checkpoint/{self.checkpoint_dir}", f'bestmodel_transformer_{self.model_type}')
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
            best_file = os.path.join(f"Saved_Models/checkpoint/{self.checkpoint_dir}", f'bestmodel_transformer_{self.model_type}')
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
            return lr * 0.2
        else:
            return lr