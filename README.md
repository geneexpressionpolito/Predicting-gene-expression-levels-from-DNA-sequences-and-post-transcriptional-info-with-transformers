# Gene Expression Prediction from Sequence

<img alt="Keras" src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"/>
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" />
<img alt="NumPy" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>

## 1. Google Drive Organization
Our work took place on the **Google Drive** environment, in order to collaborate each other and keep the development and the cooperation fast and easy.
In particular, we exploited the **Google Colab notebooks**, because  everyone  can  modify  them  once  they’re  shared  on the  drive,  and  thanks  to  ```import-ipynb``` we can import the notebooks in the workflow of other notebooks, increasing the modularity and the reproducibility of our code. 

All the notebooks are contained in the ```Colab``` subfolder of our drive.
The folder ```/Classes``` contains the model classes and other useful classes like the ```DataManager```. Those notebooks are exploited by the several  workflow  notebooks  which  are  located in ```/WORKFLOW_GPU``` and ```/WORKFLOW_TPU```
The most important notebooks are:
- ```DataManager.ipynb```: used to manage all the datasets;
- ```CNN1D.ipynb```:  contains  a  class  that  manages  all  our  convolutional  solutions  (DivideEtImpera  included),  and  areproduction of the ”Xpresso” original neural network;
- ```BioLSTM.ipynb```: contains a class that manages our main LSTM solutions;
- ```Transformer.ipynb```: contains a class that handle our transformer ssolutions. 

The  ```Workflow```  notebooks  are  organized  as  follows: 
- at the  beginning  we  have  a  call  to  the  DataManager  in  order to  retrieve  useful  data
- then, we  choose  the  best  suited model  for  our  research  choosing  between  the  ones  available in ```projCNN1D```, ```projTransformer```, and ```BioLSTM``` class.

The  construction  of  our  workflows  gave  us  the  possibility to  implement  any  kind  of  Deep  Neural  Network  and  change parameters in matters of seconds. 
The  workflow  name  suggests  the  model  and  the  data  used 
```python
{
    "P" : "Promoter", 
    "H" : "Half−Life", 
    "T" : "TranscriptionFactors",
    "M" : "MicroRNA"
}
```
Example:
`Workflow_CNN1D_PH_GPU` uses one or more models defined in CNN1D, it takes Promoters  and  Halflife features, and deploy the models on GPU.

## DeepLncLoc from scratch
This is a notebook that reimplement from scratch the full pipeline of the DeepLncLoc embedding and can be found in the ```./Colab/varie``` folder.
It can be used to retrain the Word2Vec model and to generate dataset with different parameters.


## 2. Classes


### 2.1 DataManager class
The **DataManager** class manages and imports the dataset in various forms, depending on the model that will be used for training.

```python
dm = DataManager(
    datadir         = datadir, 
    transformer     = False, 
    micro           = False, 
    tf              = False, 
    datadir_micro   = "Dataset/microRNA FINALE", 
    datadir_tf      = "Dataset/dataset_aumentati", 
    remove_indicted = False,
    DeepLncLoc      = False,
)

dataset = dm.get_train(
    np_format  = True,  # boolean to transform in numpy
    translated = False, # sequence data translated into categorical (e.g. for transformers)
    micro      = False  # if True, it returns also microRNA
)

dataset = dm.get_train(
    np_format  = True, 
    translated = False,
    micro      = False 
)

dataset = dm.get_train(
    np_format  = True, 
    translated = False,
    micro      = False 
)
```

| Parameter | Description | Values |
| ------ | ------ | ------ |
| **datadir** | *choose the directory of the dataset* | str |
| **transformer** | *sequence data translated into categorical* | boolean |
| **micro** | *retrieve also microRNA data* | boolean |
| **tf** | *retrieve also Transcription Factor data* | boolean |
| **datadir_micro** | *directory to the microRNA data* | str |
| **datadir_tf** | *directory to the Transcription Factor data* | str|
| **remove_indicted** | *keep only sequences of lenght 20_000* | boolean |
| **DeeplncLoc** | *retrieve DeepLncLoc embedded sequences (e.g. for our LSTM solution)* | boolean |

The DataManager reads the different `.h5 format` datasets from the `Dataset/` folder.

The standard data are taken from (Xpresso Original Dataset):
```
- "Dataset/pM10Kb_1KTest/train.h5"
- "Dataset/pM10Kb_1KTest/valid.h5"
- "Dataset/pM10Kb_1KTest/test.h5"
```

The data integrated with microRNA are taken from:
```
- "Dataset/microRNA FINALE/train.h5"
- "Dataset/microRNA FINALE/valid.h5"
- "Dataset/microRNA FINALE/test.h5"
```

The data integrated with Transcription Factors are taken from:
```
- "Dataset/dataset_aumentati/train_tf.h5"
- "Dataset/dataset_aumentati/validation_tf.h5"
- "Dataset/dataset_aumentati/test_tf.h5"
```

The standard translated data (for Transformers) are taken from:
```
- "Dataset/dataset_aumentati/translated_transformers.h5"
```

The standard translated data (for Transformers) integrated with Transcription Factors are taken from:
```
- "Dataset/dataset_aumentati/translated_transformers_tf.h5"
```


### 2.2 projCNN1D class

The **projCNN1D** class exposes the models based on *Convolutional 1D Neural Networks*. 
```python
model = projCNN1D(
    checkpoint_dir     = "",
    model_type         = "Xpresso",
    n_epochs           = 10, 
    batch_size         = 32, 
    learning_rate      = 5e-4,
    momentum           = 0.9,
    CNN_input          = (10_500, 4),
    miRNA_input        = (2064,),
    lr_reduction_epoch = None,
    dropout_rate       = 0.4,
    shuffle            = True,
    logdir             = None,
    patience           = 30,
    opt                = "SGD",
    loss               = "mse"
)
```

| Parameter | Description | Values |
| ------ | ------ | ------ |
| **model_type** | *choose the correct model for each dataset* | 'Xpresso', 'Xpresso_nohalf', 'Xpresso_TF', 'Xpresso_micro', 'Xpresso_ort', 'DivideEtImpera', 'DivideEtImpera_TF', 'DivideEtImpera_onlyPromo' |
| **n_epochs** | *number of epochs* | int |
| **batch_size** | *size of the batch* | int |
| **CNN_input** | *shape of the sequence data* | (n,m) |
| **dropout_rate** | *select the dropout for the Dense layers* | float |
| **logdir** | *tensorboard directory, if this parameter is set to None, the tensorboard_callback will not be used by the model* | str|
| **checkpoint_dir** | *directory in which is saved the best model* | str|
| **patience** | *patience for the earlystopping* | int |
| **learning_rate** | *learning rate* | float |
| **lr_reduction_epoch** | *milestone where lr_scheduler is invoked to reduce learning rate* | int |
| **momentum** | *momentumtum for the SGD optimizer* | float |
| **shuffle** | *data shuffle* | boolean |
| **opt** | *choose the optimizer* | 'SGD', anything else(='Adam') |

-----------------------

| model_type | Description | 
| ------ | ------ |
| **Xpresso** | *official Xpresso model* |
| **Xpresso_nohalf** | *Xpresso with input only promoters* |
| **Xpresso_TF** | *Xpresso with Promoter, Half-life and TF as input* |
| **Xpresso_micro** | *Xpresso with Promoter, Half-life and microRNA as input* |
| **Xpresso_ort** | *Basically it is equal to Xpresso, but with an orthogonal initialization of the weight* |
| **DivideEtImpera** | *DivideEtImpera with only Promoters and Half-life data as input* |
| **DivideEtImpera_TF** | *DivideEtImpera model with also TF data as input* |
| **DivideEtImpera_onlyPromo** | *DivideEtImpera with only promoters as input* |


### 2.3 BioLSTM class

The **BioLSTM** class exposes the models based on *LSTM block(s)*.
```python
model = BioLSTM(
    model_type = 'classic', 
    n_epochs   = 50, 
    batch_size = 128, 
    timestep   = 210, 
    features   = 64,
    datadir    = 'Dataset/embedded_data', 
    opt        = 'adam', 
    lr         = 3e-4
)
```


| Parameter | Description | Values |
| ------ | ------ | ------ |
| **model_type** | *choose the correct model for each dataset* | 'classic', 'tf', 'only promoter' |
| **n_epochs** | *number of epochs* | int |
| **batch_size** | *size of the batch* | int |
| **timestep** | *length of the timestep of the sequence data* | int |
| **features** | *length of the features for each timestep in the sequence data* | int |
| **datadir** | *directory in which is saved the best model* | str|
| **opt** | *optimizer used by the model* | 'adam', 'sgd' |
| **lr** | *learning rate* | float |

------------------------

| model_type | Description | 
| ------ | ------ |
| **classic** | *BioLSTM model with input promoters and Half-life data* |
| **tf** | *BioLSTM with Promoter, Half-life data and TF as input* |
| **only promoter** | *BioLSTM with promoters as input* |





### 2.4 projTFNet class

The **projTFNet** class exposes the *Fully Connected Neural Network* to make the regression only on transcription factor data.
```python
model = projTFNet(
    checkpoint_dir     = "",
    model_type         = "TF",
    n_epochs           = 10, 
    batch_size         = 32, 
    learning_rate      = 5e-4,
    momentum           = 0.9,
    lr_reduction_epoch = None,
    shuffle            = True,
    logdir             = None,
    patience           = 30
)
```

| Parameter | Description | Values |
| ------ | ------ | ------ |
| **model_type** | *choose the correct model for each dataset* | 'TF' |
| **n_epochs** | *number of epochs* | int |
| **batch_size** | *size of the batch* | int |
| **CNN_input** | *shape of the sequence data* | (n,m) |
| **dropout_rate** | *select the dropout for the Dense layers* | float |
| **logdir** | *tensorboard directory* | str|
| **checkpoint_dir** | *directory in which is saved the best model* | str|
| **patience** | *patience for the earlystopping* | int |
| **learning_rate** | *learning rate* | float |
| **lr_reduction_epoch** | *milestone where lr_scheduler is invoked to reduce learning rate* | int |
| **momentum** | *momentumtum for the SGD optimizer* | float |
| **shuffle** | *data shuffle* | boolean |

### 2.5 projTransformer class

The **projTransformer** (```Classes/Transformer.ipynb```) class exposes the *Transformers-like models*.

```python
model = projTransformer(
    checkpoint_dir     = "",
    model_type         = "best",
    n_epochs           = 300, 
    batch_size         = 32, 
    learning_rate      = 1e-4,
    momentum           = 0.9,
    maxlen             = 10500,
    embed_dim          = 32,
    num_heads          = 4,
    ff_dim             = 64,
    vocab_size         = 5,
    dense              = 64,
    lr_reduction_epoch = None,
    dropout_rate       = 0.1,
    t_rate             = 0.1,
    patience           = 20,
    optimizer          = "SGD",
    warmup_steps       = 8000,
    shuffle            = True,
    loss               = "mse",
    logdir             = None
)
```

| Parameter | Description | Values |
| ------ | ------ | ------ |
| **model_type** | *choose the correct model for each dataset* | 'DeepLncLoc', 'DeepLncLoc_TF', 'DeepLncLoc_Promoter' |
| **n_epochs** | *number of epochs* | int |
| **batch_size** | *size of the batch* | int |
| **maxlen** | *length of the sequence data* | int |
| **dropout_rate** | *select the dropout for the Dense layers* | float |
| **logdir** | *tensorboard directory* | str|
| **checkpoint_dir** | *directory in which is saved the best model* | str|
| **patience** | *patience for the earlystopping* | int |
| **learning_rate** | *learning rate* | float |
| **lr_reduction_epoch** | *milestone where lr_scheduler is invoked to reduce learning rate* | int |
| **momentum** | *momentum for the SGD optimizer* | float |
| **shuffle** | *data shuffle during train* | boolean |
| **embed_dim** | *dimension of the embedding (depth of w2v)* | int |
| **num_heads** | *number of heads* | int |
| **ff_dim** | *number of neurons of the feed forward network* | int |
| **vocab_size** | *used in word2vec* | int |
| **dense** | *number of neurons of Dense layers* | int |
| **t_rate** | *dropout rate used in the dense layers of transformers blocks* | float |
| **warmup_steps** | *steps to warmup the learning rate in original transformer scheduler/optimizer* | int |
| **loss** | *loss used* | instance of tf.keras losses (or just string name e.g. "mse")|
| **optimizer** | *optimizer used* | 'SGD', 'Adam', 'Adadelta', 'Adamax', 'Original' |

-------------------

| model_type | Description | 
| ------ | ------ |
| **DeepLncLoc** | *DeepLncLoc Transformer* |
| **DeepLncLoc_TF** | *DeepLncLoc Transformer and as input promoters, Half-Life and TF data* |
| **DeepLncLoc_onlyPromo** | *DeepLncLoc Transformer, but only promoters as input* |


## 3. Workflow Example (GPU)

### 3.1 Prerequisites

1. Connect your Google Drive account and allow to import all the notebooks (e.g. DataManager, CNN1D)
2. Once you have been added to the drive repository, in order to let the import to work properly, you need to add a shortcut of the repository in the main directory of your drive, otherwise the `%cd` command will fail and you will not be able to run our notebook properly.


```python
!pip install import-ipynb
import import_ipynb

import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

%cd "drive/MyDrive/Bionformatics_Project/Colab"
```

### 3.2 Classes import:

```python
from Classes.DataManager import DataManager
from Classes.Transformer import projTransformer
from tensorflow import keras
import numpy as np

%load_ext tensorboard
```

### 3.3 Retrieve training data from the Data Manager:

```python
dm = DataManager(transformer=False, micro=False)

X_trainhalflife, X_trainpromoter, y_train, _, _                 = dm.get_train(True, False, False)
X_validationhalflife, X_validationpromoter, y_validation, _, _  = dm.get_validation(True, False, False)
X_testhalflife, X_testpromoter, y_test, _, _                    = dm.get_test(True, False, False)
```

```python
leftpos  = 3_500          
rightpos = 13_500         
maxlen   = rightpos-leftpos
```


```python
X_trainpromoter      = X_trainpromoter[:, leftpos:rightpos, :]
X_validationpromoter = X_validationpromoter[:, leftpos:rightpos, :]
X_testpromoter       = X_testpromoter[:, leftpos:rightpos, :]
```

### 3.4 Train the model:

```python
model_type     = "Xpresso"
checkpoint_dir = "myFirstTrain/"
logdir         = "tensorboardDir/"

net = projCNN1D(
    checkpoint_dir =checkpoint_dir, 
    model_type     = model_type, 
    n_epochs       = 300, 
    batch_size     = 256, 
    learning_rate  = 5e-4, 
    CNN_input      = (maxlen, 4), 
    dropout_rate   = 0.5, 
    logdir         = logdir
)
        
net.train_model(
    [X_trainpromoter, X_trainhalflife],
    y_train, 
    [X_validationpromoter, X_validationhalflife], 
    y_validation
)
```

### 3.5 Evaluate the model:

#### 3.5.1 At the last epoch:
```python
net.evaluate([X_testpromoter, X_testhalflife], y_test)
```
#### 3.5.2 At the epoch with lowest validation loss:

```python
net.evaluate_best([X_testpromoter, X_testhalflife], y_test)
```

### 3.6 Visualize the distribution of the predictions with respect to the target:

```python
net.plot_train()
net.plot_r2([X_testpromoter, X_testhalflife], y_test)
net.plot_kde([X_testpromoter, X_testhalflife], y_test)
```
- plot_r2 shows a scatter plot of the distance of the prediction from the true values
- plt_kde match the kernel density distribution of the predicted values and the targets

## 4. Workflow Example (TPU)
### 4.1 Prerequisites:
The TPU workflow configuration is barely the same of the GPU one, you have to just to be aware of defining the model into the TPU scope, like here:

```
import tensorflow as tf
tf.keras.backend.clear_session()

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.experimental.TPUStrategy(resolver)


with strategy.scope():
    net = projTransformer(checkpoint_dir="batch128/",model_type=model_type, n_epochs=500, 
                      batch_size=256, learning_rate=5e-4, lr_reduction_epoch=200, 
                      maxlen=maxlen, embed_dim=64, num_heads=4, ff_dim=384, dense=100, 
                      dropout_rate=0.1, optimizer="Adam", warmup_steps=4_000, patience=20)
```

### 4.2 Considerations:
- You do not need to call `net.train_model()` into the TPU scope, and we suggest to call it outside, maybe in the following notebook cell;
- You may need to increase the `batch_size` in order to exploit a higher parallelization. A rule of thumb suggests to multuply by an 8 factor the `batch_size`. In this project we used the same `batch_size` that we used with GPU models in order to avoid a further hyperparameter research, giving the fact that the TPU models gives the same results when deployed on GPU if they share the same hyperparameters. Obviously, even if the `batch_size` is the same, the model will train faster on TPU.
- Due to the fact that the model runs on the TPU cloud, we cannot exploit some callbacks like `ModelCheckpoint`. So we decided to use a parameter of `EarlyStopping` which allow us to restore the net's weight of of the epoch with mininum validation loss. As a consequence, once the training is completed, `net.evaluate()` and `net.evaluate_best()` will return the same value. Moreover, net.model.save() will not work for the same reasons.
- `Tensorboard` cannot be exploited when you run on TPU, for the same reason we cited before (TPU models runs on cloud)
- `BioLSTM` has no the TPU counterpart because it is alredy fast when deployed on GPU, so it would have been useless to implement it.
- `remove_indicted` option work only on for the Xpresso dataset. If you want to train our transformer with the transcription factor dataset you will need to set `vocab_size = 5`.


## 5. Further Considerations:
When we started this project, the latest tensorflow version was 2.4.2, but after two months it changed to 2.5.x, so if you have problem in loading our `Saved_Models`, you may need to force the installation of the correct tensorflow library.

```
!pip install tensorflow==2.4.2

```

## 6. Best HyperParameters:

    BioLSTM (BioLSTM in BioLSTM.ipynb):
        model_type = "classic"
        n_epochs = 140
        batch_size = 128
        learning_rate = 3e-4
        patience = 15
        optimizer = "Adam"
        timestep = 210
        features = 64

    Our_Transformer (projTransformer in Transformer.ipynb):
        model_type = "best"
        n_epochs = 300
        batch_size = 256
        learning_rate = 1e-3
        patience = 20
        optimizer = "SGD"
        lr_reduction_epoch = 60
        embed_dim = 32
        num_heads = 1
        vocab_size = 4 (#increase this value if you feed the network with more than 4 symbols)
        ff_dim = 64
        dense = 64
        dropout_rate = 0.1
        t_rate = 0.1
        momentum = 0.9
    
    DivideEtImpera (projCNN1D in CNN1D.ipynb):
        model_type = "DivideEtImpera"
        n_epochs = 300
        batch_size = 256
        learning_rate = 1e-3
        patience = 15
        optimizer = "SGD"
        lr_reduction_epoch = 70
        momentum = 0.9
    
    Transformer DeepLncLoc (projTransformer in Transformer.ipynb):
        model_type = "DeepLncLoc"
        n_epochs = 300
        batch_size = 256
        learning_rate = 5e-4
        patience = 20
        optimizer = "Adam"
        lr_reduction_epoch = None
        embed_dim = 64
        num_heads = 4
        ff_dim = 384
        dense = 100
        dropout_rate = 0.1

    
### N.B:
### The parameters that are not present in these lists have to be assumed as the predefined parameters of the respective class.

