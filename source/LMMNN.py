import os, random, time, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Add, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from scipy import integrate
from scipy.stats import norm
import lifelines

import logging
tf.get_logger().setLevel(logging.ERROR)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)

def data_to_tensors(X, y, Z=None):    
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)    
    if len(X.shape) == 1: X = tf.reshape(X, (-1,1))
    if len(y.shape) == 1: y = tf.reshape(y, (-1,1))
    if Z is not None:
        Z = tf.convert_to_tensor(Z, dtype=tf.float32)
        return [X, y, Z]
    else:
        return [X, y]

class LMMNN:
    def __init__(self, n_clusters, n_features, num_nodes=None, activation=None):
        self.n_clusters = n_clusters
        self.n_features = n_features        
        self.log_sig2e = tf.Variable(0.0, trainable=True, name='log_sig2e')
        self.log_sig2v = tf.Variable(0.0, trainable=True, name='log_sig2v')                
        self.dispersion_params = [self.log_sig2e, self.log_sig2v]
        self.base_loss = lambda y_true, y_pred: nll_normal(y_true, y_pred, self.log_sig2e)
        self.build_model(num_nodes, activation)
    def build_model(self, num_nodes, activation):
        input_X = Input(shape=(self.n_features,), name='input_X', dtype='float32')
        input_Z = Input(shape=(self.n_clusters,), name='input_Z', dtype='float32')
        if num_nodes is not None:
            for k, num_node in enumerate(num_nodes):
                if k==0: m = Dense(num_node, activation=activation)(input_X)
                else: m = Dense(num_node, activation=activation)(m)
            mu_mar = Dense(1, activation='linear')(m)
        else: mu_mar = Dense(1, activation='linear')(input_X)        
        zv = Dense(1, activation='linear', use_bias=False, name='zv')(input_Z)
        # compute mean predictor
        mu_con = Add()([mu_mar, zv])
        self.mean_model_mar = Model(inputs=[input_X],          outputs=mu_mar, name='mean_model_mar')
        self.mean_model_con = Model(inputs=[input_X, input_Z], outputs=mu_con, name='mean_model_con')
    def predict(self, X, Z):
        pred = np.exp(self.mean_model_con([X, Z])) - 0.5
        return np.where(pred<0, 1e-3, pred)
    def initialize(self, train_data, valid_data, sig2e_init=None, sig2v_init=None):
        X_train, y_train, Z_train = train_data
        X_valid, y_valid, Z_valid = valid_data
        self.sample_size = y_train.shape[0]
        if sig2e_init is not None: self.log_sig2e.assign(tf.math.log(sig2e_init))
        if sig2v_init is not None: self.log_sig2v.assign(tf.math.log(sig2v_init))
        self.cluster_sizes = tf.reshape(tf.reduce_sum(Z_train, axis=0),(-1,1))
        fX_train = self.mean_model_mar([X_train])
        vpred = (tf.transpose(Z_train)@(y_train-fX_train))/(self.cluster_sizes+tf.exp(self.log_sig2e-self.log_sig2v))
        layer = self.mean_model_con.get_layer(name='zv')
        layer.set_weights([vpred])
        self.train_losses, self.valid_losses = [], []
        self.history = {}
        self.history['time']  = [0]
        self.history['sig2e'] = [tf.exp(self.log_sig2e)]
        self.history['sig2v'] = [tf.exp(self.log_sig2v)]
    def compute_mlik(self, y_true, y_pred, Z):
        errs = y_true - y_pred
        ycov = tf.exp(self.log_sig2e)*tf.eye(y_true.shape[0]) + Z@tf.transpose(Z)*tf.exp(self.log_sig2v)
        loss = (tf.transpose(errs)@tf.linalg.solve(ycov, errs) + tf.linalg.slogdet(ycov)[1])/y_true.shape[0]/2
        return loss
    def train_mlik(self, train_data, valid_data, optimizer, batch_size, max_epochs, patience, sig2e_init, sig2v_init, shuffle=False):
        train_data = data_to_tensors(*train_data)
        valid_data = data_to_tensors(*valid_data)
        X_train, y_train, Z_train = train_data
        X_valid, y_valid, Z_valid = valid_data
        self.initialize(train_data, valid_data, sig2e_init, sig2v_init)
        if not shuffle: train_batch = tf.data.Dataset.from_tensor_slices((X_train, y_train, Z_train)).batch(batch_size)
        patience_count = 0
        min_valid_loss = np.infty
        for epoch in range(max_epochs):
            start_time  = time.time()
            if shuffle: train_batch = tf.data.Dataset.from_tensor_slices((X_train, y_train, Z_train)).shuffle(self.sample_size).batch(batch_size)
            for step, (X_batch, y_batch, Z_batch) in enumerate(train_batch):
                with tf.GradientTape() as tape:
                    train_loss = self.compute_mlik(y_batch, self.mean_model_mar([X_batch]), Z_batch)
                gradients = tape.gradient(train_loss, self.mean_model_mar.trainable_weights+self.dispersion_params)
                optimizer.apply_gradients(zip(gradients, self.mean_model_mar.trainable_weights+self.dispersion_params))
                self.train_losses.append(train_loss)                    
            fX_train = self.mean_model_mar([X_train])
            vpred = (tf.transpose(Z_train)@(y_train-fX_train))/(self.cluster_sizes+tf.exp(self.log_sig2e-self.log_sig2v))
            layer = self.mean_model_con.get_layer(name='zv')
            layer.set_weights([vpred])
            self.history['sig2e'].append(tf.exp(self.log_sig2e))
            self.history['sig2v'].append(tf.exp(self.log_sig2v))
            self.history['time'].append(self.history['time'][-1]+time.time()-start_time)
            valid_loss = self.compute_mlik(y_valid, self.mean_model_mar([X_valid]), Z_valid)
            self.valid_losses.append(valid_loss)
            if valid_loss > min_valid_loss: patience_count += 1
            else: min_valid_loss = valid_loss
            if patience_count == patience: break