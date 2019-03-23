# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:33:01 2019

@author: Ethan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

folder = 'mercari-price-suggestion-challenge'
train_data = pd.read_table(os.path.join(folder, 'train.tsv'))
test_data = pd.read_table(os.path.join(folder, 'test.tsv'))
labels = list(train_data.columns)

train_sample = len(train_data)
test_sample = len(test_data)
y_train = train_data['price']
y_train = np.array(y_train).reshape(-1, 1)
train_data = train_data.drop('price', axis = 1)
# onehot categorical variable
def fillnan(df):
    df.brand_name = df.brand_name.fillna(value = 'Unknown')
    df.category_name = df.category_name.fillna(value ='missing')
    df.item_description = df.item_description.fillna(value = 'No description')
    return df
train_data = fillnan(train_data)
test_data = fillnan(test_data)
full_data = pd.concat((train_data, test_data), axis = 0)

train_X = {}
test_X = {}

train_X['item_condition_id'] = np.array(train_data.item_condition_id)
train_X['shipping'] = np.array(train_data.shipping)

test_X['item_condition_id'] = np.array(test_data.item_condition_id)
test_X['shipping'] = np.array(test_data.shipping)

def EncodeLabels(encodeLabel1, encodeLabel2):
    encoder1 = LabelEncoder()
    encoder2 = LabelEncoder()
    encoder1.fit(encodeLabel1)
    encoder2.fit(encodeLabel2)
    return (encoder1.transform(encodeLabel1.iloc[:train_sample]), 
            encoder1.transform(encodeLabel1.iloc[train_sample:]),
            encoder2.transform(encodeLabel2.iloc[:train_sample]),
            encoder2.transform(encodeLabel2.iloc[train_sample:]))
            
(train_X['category_name'], test_X['category_name'], train_X['brand_name'], test_X['brand_name']) = EncodeLabels(full_data['category_name'],full_data['brand_name'])
brand_num = np.max([np.max(train_X.get('brand_name')), np.max(test_X.get('brand_name'))]) 
ctg_num = np.max([np.max(train_X.get('category_name')),np.max(test_X.get('category_name'))])                                                                                                             
 

def tokenizer(column_name):
    tokenize = Tokenizer(num_words = 20000)
    tokenize.fit_on_texts(list(train_data[column_name].str.lower()))
    train_sequences = tokenize.texts_to_sequences(list(train_data[column_name].str.lower()))
    test_sequences = tokenize.texts_to_sequences(list(test_data[column_name].str.lower()))
    max_len = 10 if column_name == 'name' else 50
    train = sequence.pad_sequences(train_sequences, maxlen = max_len)
    test = sequence.pad_sequences(test_sequences, maxlen = max_len)
    del tokenize
    return train, test

train_X['name'], test_X['name'] = tokenizer(column_name = 'name')
train_X['item_description'], test_X['item_description'] = tokenizer(column_name = 'item_description')


# Construct a Multi-input nearaul network
from keras.layers import Embedding, Dense, SeparableConv1D, GlobalAveragePooling1D, BatchNormalization, Input
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.models import Model
import keras
def BuildModel(lr = 0.001, decay = 0.0):
    
    brand_name = Input(shape=[1], name = 'brand_name')
    category_name = Input(shape=[1], name = 'category_name')
    item_condition_id = Input(shape=[1], name = 'item_condition_id') 
    item_description = Input(shape=(50,), name = 'item_description')
    name = Input(shape=(10,), name = 'name')    
    shipping = Input(shape=[1], name = 'shipping')
    
    em_ctg = Embedding(ctg_num + 1, 10)(category_name)
    em_brand = Embedding(brand_num + 1, 10)(brand_name)    
    em_desc = Embedding(20000, 100)(item_description)
    em_name = Embedding(20000, 20)(name)
    
    
    x = SeparableConv1D(64,3,activation = 'relu')(em_name)
    x = GlobalAveragePooling1D()(x)
    
    y = SeparableConv1D(64, 6, activation = 'relu')(em_desc)
    y = SeparableConv1D(128,6, activation = 'relu')(y)
    y = layers.Dropout(0.3)(y)
    y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    
    main = layers.concatenate([layers.Flatten()(em_brand),
                               layers.Flatten()(em_ctg),
                               item_condition_id,
                               y,
                               x,
                               shipping,])
    dense = Dense(64, activation = 'relu')(main)
    dense = Dense(32, activation = 'relu')(dense)
    dense = Dense(1, activation = 'linear')(dense)
    
    model = Model([brand_name, 
                   category_name, 
                   item_condition_id,
                   item_description,
                   name,
                   shipping,], dense)
    
    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="mse", optimizer=optimizer)
    
    return model

model = BuildModel()
model.summary()

callback_list = [keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 1),
                 keras.callbacks.ModelCheckpoint(monitor = 'val_loss',
                                                 save_weights_only = True,
                                                 filepath = folder + '/MultiinputModel.h5')]


model.fit(train_X, y_train,
          epochs = 100,
          batch_size = 1024,
          validation_split = 0.2,
          callbacks = callback_list)

# Final prediction on the test data
pred_price = model.predict(test_X, batch_size = 1024, verbose = 1)



