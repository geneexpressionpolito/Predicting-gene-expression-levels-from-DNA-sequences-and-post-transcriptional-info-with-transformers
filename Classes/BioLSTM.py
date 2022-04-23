import numpy as np
import gc
import os,logging,pickle,random
from matplotlib import pyplot
import pandas as pd
from scipy import stats
import keras
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, LSTM, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class BioLSTM:
    def __init__(self, model_type="tf", n_epochs=50, batch_size=128, timestep=210, features=64, datadir='Dataset/embedded_data', opt='adam', lr=3e-4):
        
        self.batch_size = batch_size
        self.model_type = model_type
        self.n_epochs = n_epochs
        self.timestep = timestep
        self.features = features
        self.datadir = datadir
        self._build_model()
        self.opt = opt
        self.lr = lr
    def _build_model(self):
           
        if self.model_type == "tf":
          input_promoter = Input(shape=(self.timestep, self.features), name='promoter')
          halflife = Input(shape=(8,), name = 'halflife')
          tf = Input(shape=(181,), name = 'tf')

          x = layers.BatchNormalization()(input_promoter)
          x = LSTM(units = 100, input_shape=(self.timestep,self.features))(x)
          #y = layers.BatchNormalization()(halflife)
          c = Concatenate()([x, halflife, tf])
          x = layers.Dense(90, activation="relu")(c)
          x = layers.Dropout(0.3)(x)
          output = layers.Dense(1, activation="linear")(x)

          self.model = Model(inputs=(input_promoter,halflife,tf), outputs=output)

        if self.model_type == "classic":
          input_promoter = Input(shape=(self.timestep, self.features), name='promoter')
          halflife = Input(shape=(8,), name = 'halflife')

          x = layers.BatchNormalization()(input_promoter)
          x = LSTM(units = 100, input_shape=(self.timestep,self.features))(x)
          #y = layers.BatchNormalization()(halflife)
          c = Concatenate()([x, halflife])
          x = layers.Dense(90, activation="relu")(c)
          x = layers.Dropout(0.3)(x)
          output = layers.Dense(1, activation="linear")(x)

          self.model = Model(inputs=(input_promoter,halflife), outputs=output)

        if self.model_type == "only promoter":
          input_promoter = Input(shape=(self.timestep, self.features), name='promoter')

          x = layers.BatchNormalization()(input_promoter)
          x = LSTM(units = 100, input_shape=(self.timestep,self.features))(x)
          x = layers.Dense(90, activation="relu")(x)
          x = layers.Dropout(0.3)(x)
          output = layers.Dense(1, activation="linear")(x)

          self.model = Model(inputs=(input_promoter), outputs=output)
          
        print(self.model.summary())
        print(f"\nParameters:\n{vars(self)}\n")

    def train_model(self, x, y, x_v=None, y_v=None):
      if self.opt == 'adam':
        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')
      if self.opt == 'sgd':
        self.model.compile(optimizer=SGD(learning_rate=self.lr, momentum=0.9), loss='mse')
      earlystop_cb = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
      check_cb = ModelCheckpoint(os.path.join(self.datadir, 'best.h5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
      history = self.model.fit(x, y, batch_size=128, epochs=self.n_epochs, verbose=1, validation_data=(x_v,y_v),callbacks=[earlystop_cb,check_cb])

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

    def evaluate_best(self, x, y):
        best_file = os.path.join(self.datadir, 'best.h5')
        self.model = load_model(best_file)
        predictions = self.model.predict(x).flatten()
        slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, y)
        print('Test R^2 = %.3f' % r_value**2)
        return r_value**2

    def plot_kde(self, x, y, TPU=False):
        if TPU is False:
            best_file = os.path.join(self.datadir, 'best.h5')
            model = load_model(best_file)
            predictions = model.predict(x).flatten()
        else:
            predictions = self.model.predict(x).flatten()
        df = pd.DataFrame({"predictions":predictions, "true":y})
        ax = sns.displot(data=df, kde=True)
        plt.xlabel('Labels')
        plt.show()
        
    def plot_r2(self, x, y, TPU=False):
        from matplotlib import cm
        if TPU == False:
            best_file = os.path.join(self.datadir, 'best.h5')
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
        plt.show()
