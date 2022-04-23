import sys, gzip, h5py, pickle, os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import csv
import seaborn as sns
import random
import json

random_seed = 43
random.seed(random_seed)

import sys
import numpy
numpy.set_printoptions(threshold=100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

datadir = "Dataset/pM10Kb_1KTest" # the one used by Xpresso

class DataManager():
    def __init__(self, 
                 datadir         = datadir, 
                 transformer     = False, 
                 micro           = False, 
                 tf              = False, 
                 datadir_micro   = "Dataset/microRNA FINALE", 
                 datadir_tf      = "Dataset/dataset_aumentati", 
                 remove_indicted = False,
                 DeepLncLoc        = False,
                 ):
        
        self.datadir         = datadir
        self.remove_indicted = remove_indicted
        self.indicted        = [577, 1494, 2729, 2829, 4095, 5038, 5340, 7804, 8445, 8532, 8557, 11040, 11369, 11638, 11650, 11785, 12216, 13851, 13927]
        self.indicted_test   = [437]
        self.X_traintf       = [0]
        self.X_validtf       = [0]
        self.X_testtf        = [0]
        self.DeepLncLoc        = DeepLncLoc

        if not DeepLncLoc:
            if tf:
                self.trainfile = h5py.File(os.path.join(datadir_tf, 'train_tf.h5'), 'r')
                self.validfile = h5py.File(os.path.join(datadir_tf, 'validation_tf.h5'), 'r')
                self.testfile  = h5py.File(os.path.join(datadir_tf, 'test_tf.h5'), 'r')
                
                self.X_trainhalflife = self.trainfile['halflife'] 
                self.X_trainpromoter = self.trainfile['promoter']
                self.X_traintf       = self.trainfile['tf']
                self.y_train         = self.trainfile['label'] 
                self.geneName_train  = self.trainfile['gene']

                self.X_validhalflife = self.validfile['halflife']
                self.X_validpromoter = self.validfile['promoter']
                self.X_validtf       = self.validfile['tf']
                self.y_validid         = self.validfile['label'] 
                self.geneName_valid  = self.validfile['gene']

                self.X_testhalflife  = self.testfile['halflife']
                self.X_testpromoter  = self.testfile['promoter']
                self.X_testtf        = self.testfile['tf']
                self.y_test          = self.testfile['label']
                self.geneName_test   = self.testfile['gene']
            
            elif micro:
            
                self.trainfile = h5py.File(os.path.join(datadir_micro, 'microRNA_train.h5'), 'r')
                self.validfile = h5py.File(os.path.join(datadir_micro, 'microRNA_val.h5'), 'r')
                self.testfile  = h5py.File(os.path.join(datadir_micro, 'microRNA_test.h5'), 'r')
                
                self.X_trainhalflife = self.trainfile['halflife'] 
                self.X_trainpromoter = self.trainfile['promoter']
                self.X_traintf       = self.trainfile['tf']
                self.X_trainmicro    = self.trainfile['micro']
                self.y_train         = self.trainfile['label'] 
                self.geneName_train  = self.trainfile['gene']

                self.X_validhalflife = self.validfile['halflife']
                self.X_validpromoter = self.validfile['promoter']
                self.X_validtf       = self.validfile['tf']
                self.X_valmicro      = self.validfile['micro']
                self.y_validid         = self.validfile['label'] 
                self.geneName_valid  = self.validfile['gene']

                self.X_testhalflife  = self.testfile['halflife']
                self.X_testpromoter  = self.testfile['promoter']
                self.X_testtf        = self.testfile['tf']
                self.X_testmicro     = self.testfile['micro']
                self.y_test          = self.testfile['label']
                self.geneName_test   = self.testfile['gene']

            else:
                self.trainfile = h5py.File(os.path.join(datadir, 'train.h5'), 'r')
                self.validfile = h5py.File(os.path.join(datadir, 'valid.h5'), 'r')
                self.testfile  = h5py.File(os.path.join(datadir, 'test.h5'),  'r')
                
                self.X_trainhalflife = self.trainfile['data'] 
                self.X_trainpromoter = self.trainfile['promoter'] 
                self.y_train         = self.trainfile['label'] 
                self.geneName_train  = self.trainfile['geneName']

                self.X_validhalflife = self.validfile['data']
                self.X_validpromoter = self.validfile['promoter']
                self.y_validid         = self.validfile['label'] 
                self.geneName_valid  = self.validfile['geneName']

                self.X_testhalflife  = self.testfile['data']
                self.X_testpromoter  = self.testfile['promoter']
                self.y_test          = self.testfile['label']
                self.geneName_test   = self.testfile['geneName']

        else:

            datadir = 'Dataset/embedded_data'

            if tf:
                trainfile = h5py.File(os.path.join(datadir, 'etrain_tf.h5'), 'r')
                validfile = h5py.File(os.path.join(datadir, 'evalidation_tf.h5'), 'r')
                testfile  = h5py.File(os.path.join(datadir, 'etest_tf.h5'), 'r')

                self.X_trainhalflife, self.X_trainpromoter, self.y_train, self.X_traintf = trainfile['halflife'], trainfile['promoter'], trainfile['label'], trainfile['tf']
                self.X_validhalflife, self.X_validpromoter, self.y_valid, self.X_validtf = validfile['halflife'], validfile['promoter'], validfile['label'], validfile['tf']
                self.X_testhalflife,  self.X_testpromoter,  self.y_test,  self.X_testtf  = testfile['halflife'],  testfile['promoter'],  testfile['label'],  testfile['tf']

                self.X_trainpromoter, self.X_validpromoter, self.X_testpromoter = np.array(self.X_trainpromoter), np.array(self.X_validpromoter), np.array(self.X_testpromoter)
                self.X_trainhalflife, self.X_validhalflife, self.X_testhalflife = np.array(self.X_trainhalflife), np.array(self.X_validhalflife), np.array(self.X_testhalflife)
                self.X_traintf,       self.X_validtf,       self.X_testtf       = np.array(self.X_traintf),       np.array(self.X_validtf),       np.array(self.X_testtf)
                self.y_train,         self.y_valid,         self.y_test         = np.array(self.y_train),         np.array(self.y_valid),         np.array(self.y_test)

            else:
                trainfile = h5py.File(os.path.join(datadir, 'etrain.h5'), 'r')
                validfile = h5py.File(os.path.join(datadir, 'evalidation.h5'), 'r')
                testfile  = h5py.File(os.path.join(datadir, 'etest.h5'), 'r')

                self.X_trainhalflife, self.X_trainpromoter, self.y_train = trainfile['halflife'], trainfile['promoter'], trainfile['label']
                self.X_validhalflife, self.X_validpromoter, self.y_valid = validfile['halflife'], validfile['promoter'], validfile['label']
                self.X_testhalflife,  self.X_testpromoter,  self.y_test  = testfile['halflife'],  testfile['promoter'],  testfile['label']

                self.X_trainpromoter, self.X_validpromoter, self.X_testpromoter = np.array(self.X_trainpromoter), np.array(self.X_validpromoter), np.array(self.X_testpromoter)
                self.X_trainhalflife, self.X_validhalflife, self.X_testhalflife = np.array(self.X_trainhalflife), np.array(self.X_validhalflife), np.array(self.X_testhalflife)
                self.y_train,         self.y_valid,         self.y_test         = np.array(self.y_train), np.array(self.y_valid), np.array(self.y_test)

        #translated versions
        if transformer:
            if not tf:
                self.dictionary = {"A":0, "C":1, "G":2, "T":3}
                transformersdata   = h5py.File(os.path.join("Dataset/translated_promoters", 'translated_transformers.h5'), 'r')
                self.X_trainpromoter_tr = transformersdata["train"]
                self.X_validpromoter_tr = transformersdata["valid"]
                self.X_testpromoter_tr  = transformersdata["test"]
            else:
                self.dictionary = {"A":0, "C":1, "G":2, "T":3}
                transformersdata   = h5py.File(os.path.join("Dataset/dataset_aumentati", 'translated_transformers_tf.h5'), 'r')
                self.X_trainpromoter_tr = transformersdata["train"]
                self.X_validpromoter_tr = transformersdata["valid"]
                self.X_testpromoter_tr  = transformersdata["test"]
        

    def get_train(self, np_format=True, translated=False, micro=False):
        
        if self.DeepLncLoc:
            return self.X_trainhalflife, self.X_trainpromoter, self.y_train, self.X_traintf
        else:
            if np_format:
                
                self.X_trainhalflife = np.array(self.X_trainhalflife)
            
                if translated:
                    
                    if self.remove_indicted:
                    
                        train_translated_no_indicted_halflife = np.delete(self.X_trainhalflife, self.indicted, 0)
                        train_translated_no_indicted_promoter = np.delete(np.array(self.X_trainpromoter_tr), self.indicted, 0)
                        train_translated_no_indicted_labels   = np.delete(np.array(self.y_train), self.indicted, 0)
                        train_translated_no_indicted_names    = np.delete(np.array(self.geneName_train), self.indicted, 0)
                        
                        return train_translated_no_indicted_halflife, train_translated_no_indicted_promoter, train_translated_no_indicted_labels, train_translated_no_indicted_names
                        
                    else:

                        train_translated_halflife = np.array(self.X_trainhalflife)
                        train_translated_promoter = self.X_trainpromoter_tr
                        train_translated_labels   = np.array(self.y_train)
                        train_translated_names    = np.array(self.geneName_train)
                        
                        return train_translated_halflife, train_translated_promoter, train_translated_labels, train_translated_names, np.array(self.X_traintf)

                elif micro:

                    return np.array(self.X_trainhalflife), np.array(self.X_trainpromoter), np.array(self.y_train), np.array(self.geneName_train), np.array(self.X_traintf), np.array(self.X_trainmicro)

                elif self.remove_indicted:

                    train_no_indicted_halflife = np.delete(self.X_trainhalflife, self.indicted, 0)
                    train_no_indicted_promoter = np.delete(np.array(self.X_trainpromoter), self.indicted)
                    train_no_indicted_labels   = np.delete(np.array(self.y_train), self.indicted)
                    train_no_indicted_names    = np.array(self.geneName_train)
                    
                    return train_no_indicted_halflife, train_no_indicted_promoter, train_no_indicted_labels, train_no_indicted_names

                else:
                    
                    return np.array(self.X_trainhalflife), np.array(self.X_trainpromoter), np.array(self.y_train), np.array(self.geneName_train), np.array(self.X_traintf)
                
            else:
                
                return self.X_trainhalflife, self.X_trainpromoter, self.y_train, self.geneName_train, self.X_traintf
    
    def get_validation(self, np_format=True, translated=False, micro=False):

        if self.DeepLncLoc:
            return self.X_validhalflife, self.X_validpromoter, self.y_valid, self.X_validtf
        else:
            if np_format:
                
                if translated:
                    
                    valid_translated_halflife_np = np.array(self.X_validhalflife)
                    valid_translated_promoter_np = self.X_validpromoter_tr
                    valid_translated_labels_np   = np.array(self.y_validid)
                    valid_translated_names_np    = np.array(self.geneName_valid)
                    
                    return valid_translated_halflife_np, valid_translated_promoter_np, valid_translated_labels_np, valid_translated_names_np, np.array(self.X_validtf)
                
                elif micro:

                    return np.array(self.X_validhalflife), np.array(self.X_validpromoter), np.array(self.y_validid), np.array(self.geneName_valid), np.array(self.X_validtf), np.array(self.X_valmicro)
                
                else:
                    
                    return np.array(self.X_validhalflife), np.array(self.X_validpromoter), np.array(self.y_validid), np.array(self.geneName_valid), np.array(self.X_validtf)
                
            else:
            
                return self.X_validhalflife, self.X_validpromoter, self.y_validid, self.geneName_valid, self.X_validtf

    def get_test(self, np_format=True, translated=False, micro=False):

        if self.DeepLncLoc:
            return self.X_testhalflife,  self.X_testpromoter,  self.y_test,  self.X_testtf
        else:
            if np_format:

                if translated:

                    if self.remove_indicted:
                    
                        train_translated_no_indicted_halflife = np.delete(self.X_testhalflife, self.indicted_test, 0)
                        train_translated_no_indicted_promoter = np.delete(np.array(self.X_testpromoter_tr), self.indicted_test, 0)
                        train_translated_no_indicted_labels   = np.delete(np.array(self.y_test), self.indicted_test, 0)
                        train_translated_no_indicted_names    = np.delete(np.array(self.geneName_test), self.indicted_test, 0)
                        
                        return train_translated_no_indicted_halflife, train_translated_no_indicted_promoter, train_translated_no_indicted_labels, train_translated_no_indicted_names

                    else:

                        test_translated_halflife_np = np.array(self.X_testhalflife)
                        test_translated_promoter_np = self.X_testpromoter_tr
                        test_translated_labels_np   = np.array(self.y_test)
                        test_translated_names_np    = np.array(self.geneName_test)
                    
                    return test_translated_halflife_np, test_translated_promoter_np, test_translated_labels_np, test_translated_names_np, np.array(self.X_testtf)

                elif micro:
                
                    return np.array(self.X_testhalflife), np.array(self.X_testpromoter), np.array(self.y_test), np.array(self.geneName_test), np.array(self.X_testtf), np.array(self.X_testmicro)
            
                else:
                
                    return np.array(self.X_testhalflife), np.array(self.X_testpromoter), np.array(self.y_test), np.array(self.geneName_test), np.array(self.X_testtf)
            
            else:
                
                return self.X_testhalflife, self.X_testpromoter, self.y_test, self.geneName_test, self.X_testtf