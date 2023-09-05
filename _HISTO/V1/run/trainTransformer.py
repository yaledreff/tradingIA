""" 
train the time series model
"""

import sys
sys.path.insert(0, '../Time-series-prediction')
sys.path.insert(1, '../utils')
sys.path.insert(2, './configs')
sys.path.insert(3, './dataset')

import argparse
import warnings
import functools
import importlib
from copy import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import joblib

# Postgresql DB
import psycopg2
from sqlalchemy import create_engine

from tfts import build_tfts_model, KerasTrainer
from read_data import DataReader, DataLoader
from prepare_data import get_idx_from_days2
from util import set_seed, compress_submitted_zip
from feature import *
warnings.filterwarnings("ignore")
np.random.seed(315)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# ----------------------------config-----------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", default='bert_v1', help="config filename")
    parser.add_argument("--seed", type=int, default=3150, required=False, help='seed')
    parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), default=False, required=False, help='debug or not')
    return parser.parse_args()


# ----------------------------data-----------------------------------

def splitDataset(dfData, testSize=.2):
    rs = ShuffleSplit(n_splits=1, test_size=testSize)
    first_index, second_index = next(rs.split(dfData)) 
    dfData1, dfData1 = dfData.iloc[first_index], dfData.iloc[second_index] 
    return dfData1, dfData1

def build_data(cfg):
    """ 
    Prepare data pipe for model
    """

    # get data from DB
    db = create_engine(cfg.db_conn_str)
    conn = db.connect()
    df = pd.read_sql(cfg.db_sql_select, conn);
    
    df['targetBuy'] = df['rProfitBuy'] + df['rSwapBuy']
    df['targetSell'] = df['rProfitSell'] + df['rSwapSell']
    
    dfNotNa = df[df['rProfitBTrigger'].notna()]
    dfCleanRow = dfNotNa[dfNotNa['epoch'] < 1690484400]
    dfClean = dfCleanRow.drop(['rProfitBuy', 'rSwapBuy', 'rProfitSell', 'rSwapSell', 'rProfitSTrigger', 'rProfitBTrigger'], axis=1)
    # Transform regression problem to binary classification problem
    dfClean['targetProfitBuy'] = dfClean['targetBuy'].apply(lambda x: 1 if x > 0 else 0)
    dfClean['targetProfitSell'] = dfClean['targetSell'].apply(lambda x: 1 if x > 0 else 0)
    # Orders are sent just after last period closure. We need to use T-1 to predict T (dataset targets have to be realigned)
    dfClean['targetProfitBuy'] = dfClean['targetProfitBuy'].shift(-1)
    dfClean['targetProfitSell'] = dfClean['targetProfitSell'].shift(-1)
    dfClean['targetSell'] = dfClean['targetSell'].shift(-1)
    dfClean['targetBuy'] = dfClean['targetBuy'].shift(-1)
    dfClean = dfClean[dfClean['targetProfitSell'].notna()]
    dfClean.set_index('epoch', inplace=True)
    # Split dataset in Train / Valid / Test datasets
    dfData = dfClean[cfg.feature_column_short + cfg.target_column]
    dfTrain, dfValid = splitDataset(dfData, testSize=.2);
    
    


    
    
    valid_df = data.iloc[valid_idx]
    return train_data_reader, train_data_loader, valid_data_reader, valid_data_loader, valid_df


# ----------------------------model-----------------------------------

class KDD(object):
    def __init__(self, train_sequence_length, predict_sequence_length) -> None:   
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length

    def __call__(self, x, **kwargs):
        _, raw, raw_long  = x  # feature is here
        raw, manual = tf.split(raw, [10, tf.shape(raw)[-1]-10], axis=-1)
        wind_speed, wind_dir, _, _, _, _, _, _, _, active_power = tf.split(raw, 10, axis=-1)
        manual = tf.where(tf.math.is_nan(manual), tf.zeros_like(manual), manual)

        day_of_week, hour_feature, minute_of_day = tf.split(raw_long, 3, axis=-1)

        hour_feature = hour_feature / 23 - 0.5
        minute_of_day = minute_of_day / 143 - 0.5
        day_of_week = day_of_week / 6 - 0.5

        _, decoder_hour_feature = tf.split(hour_feature, [self.train_sequence_length, self.predict_sequence_length], axis=1)       
        _, decoder_minute_feature = tf.split(minute_of_day, [self.train_sequence_length, self.predict_sequence_length], axis=1)
        _, decoder_day_feature = tf.split(day_of_week, [self.train_sequence_length, self.predict_sequence_length], axis=1)

        encoder_features = tf.concat([wind_speed, wind_dir], axis=-1)        
        decoder_features = tf.concat([decoder_hour_feature, decoder_minute_feature, decoder_day_feature], axis=-1)
        decoder_features = tf.cast(decoder_features, tf.float32)
        print('Feature shape', encoder_features.shape, decoder_features.shape)
        return active_power, encoder_features, decoder_features


def build_model(use_model, train_sequence_length, predict_sequence_length=288, target_aggs=1, short_feature_nums=10, long_feature_nums=1):
    inputs = (
        Input([1]),
        Input([train_sequence_length, short_feature_nums]),  # raw feature numbers
        Input([train_sequence_length+predict_sequence_length, long_feature_nums])  # long feature
        )
    teacher_inputs = Input([predict_sequence_length//target_aggs, 1])

    ts_inputs = KDD(train_sequence_length, predict_sequence_length)(inputs)
    outputs = build_tfts_model(
        use_model=use_model, 
        predict_sequence_length=predict_sequence_length//target_aggs, 
        custom_model_params=cfg.custom_model_params)(ts_inputs, teacher_inputs)

    model = tf.keras.Model(inputs={'inputs':inputs, 'teacher': teacher_inputs}, outputs=outputs)
    return model


def custom_loss(y_true, y_pred):   
    true, mask = tf.split(y_true, 2, axis=-1)   
    mask = tf.cast(mask, dtype=tf.float32)  
    true *= mask
    y_pred *= mask
    rmse_score = tf.math.sqrt(tf.reduce_mean(tf.square(true - y_pred)) + 1e-9)
    return rmse_score


# ----------------------------train-----------------------------------

def run_train(cfg):    
    train_data_reader, train_data_loader, valid_data_reader, valid_data_loader, valid_df = build_data(cfg) 
    print(len(train_data_reader), len(valid_data_reader), len(valid_df))  

    build_model_fn = functools.partial(
        build_model, 
        cfg.use_model, 
        train_sequence_length=cfg.train_sequence_length, 
        predict_sequence_length=cfg.predict_sequence_length, 
        target_aggs=cfg.target_aggs, 
        short_feature_nums=len(cfg.feature_column_short), 
        long_feature_nums=len(cfg.feature_column_long)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate = cfg.fit_params['learning_rate'])
    loss_fn = custom_loss  

    trainer = KerasTrainer(build_model_fn, loss_fn=loss_fn, optimizer=optimizer, strategy=None)
    trainer.train(train_data_loader, valid_dataset=valid_data_loader, **cfg.fit_params)  
    trainer.save_model(
        model_dir=cfg.model_dir + '/checkpoints/{}_{}'.format(cfg.use_model, args.seed), 
        checkpoint_dir=cfg.checkpoint_dir+'/nn_{}.h5'.format(cfg.use_model)
    )
    compress_submitted_zip(res_dir='../inference', output_dir='../../weights/result')  # save to submit


if __name__ == '__main__':
    args = parse_args()
    cfg = copy(importlib.import_module(args.config).cfg)    
    set_seed(args.seed)

    cfg.fit_params = {
        'n_epochs': 2,
        'batch_size': 32,
        'learning_rate': 5e-3,
        'verbose': 1,
        'checkpoint': ModelCheckpoint(
            cfg.checkpoint_dir+'/nn_kdd_{}.h5'.format(cfg.use_model), 
            monitor='val_loss', 
            save_weights_only=True, 
            save_best_only=False, 
            verbose=1),      
    }
    
    run_train(cfg)
