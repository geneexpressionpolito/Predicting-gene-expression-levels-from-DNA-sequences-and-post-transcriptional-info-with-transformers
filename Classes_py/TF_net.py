import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import stats
from keras.models import Model, load_model
import numpy as np
import datetime, os
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass


class projTFNet:
    def __init__(   self,
                    checkpoint_dir="",
                    model_type="TF",
                    n_epochs=10, 
                    batch_size=32, 
                    learning_rate=5e-4,
                    momentum=0.9,
                    lr_reduction_epoch=None,
                    shuffle=True,
                    logdir=None,
                    patience=30):
        
        self.checkpoint_dir=checkpoint_dir
        self.model_type=model_type
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.lr_reduction_epoch = lr_reduction_epoch
        self.shuffle=shuffle
        self.logdir=logdir
        self.patience=patience

        self._build_model()

    def _build_model(self):
        if self.model_type == "TF":
            input1 = layers.Input(shape=(181))
            x = layers.Dense(64, activation="relu")(input1)
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


    def train_model(self, x_train, y_train, x_val=None, y_val=None, TPU=False):
        #train test split
        if x_val is None:
            x_train, y_train, x_val, y_val = self._split_validation_data(x_train, y_train, 0.1)
        #optimizer
        # optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate, momentum=self.momentum)
        self.model.compile(optimizer=optimizer, loss='mae')
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
        if self.logdir is not None:
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
            best_file = os.path.join(f"Dataset/checkpoint/{self.checkpoint_dir}", f'bestmodel_CNN1D_{self.model_type}')
            model = load_model(best_file)
            predictions = model.predict(x).flatten()
        else:
            predictions = self.model.predict(x).flatten()
        slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, y)
        print('Test R^2 = %.3f' % r_value**2)
        return r_value**2

    def plot_kde(self, x, y, TPU=False):
        if TPU is False:
            best_file = os.path.join(f"Dataset/checkpoint/{self.checkpoint_dir}", f'bestmodel_CNN1D_{self.model_type}')
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
            best_file = os.path.join(f"Dataset/checkpoint/{self.checkpoint_dir}", f'bestmodel_CNN1D_{self.model_type}')
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