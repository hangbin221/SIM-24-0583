import os, math, random, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
# ----------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _ = torch.manual_seed(seed)
# ----------------------------------------------------------
class HMixNet(nn.Module):
    # ----------------------------------------------------------
    def __init__(self, n_clusters, n_features, num_nodes, activation, batch_norm=False, dropout=None, device=None):
        """
        n_clusters : number of clusters
        n_features : number of input features in X
        num_nodes  : list of number of nodes in each hidden layer
        activation : nn.activation()
        """
        super(HMixNet, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.build_model(num_nodes, activation, batch_norm, dropout)
        self.device = device
        self.to(self.device)
    # ----------------------------------------------------------
    def build_model(self, num_nodes, activation, batch_norm, dropout):
        """
        exp_xb: X -> neural network models -> 1d array -> exponent
        exp_zv: Z -> 1d array (log-random-effects) -> exponent
        """
        layers = []
        if batch_norm: layers.append(nn.BatchNorm1d(self.n_features))
        in_dim = self.n_features
        for k, out_dim in enumerate(num_nodes):
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            layers.append(activation)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1, bias=True))
        self.model_mar = nn.Sequential(*layers)
        self.model_con = nn.Linear(self.n_clusters, 1, bias=False)
    def forward_x(self, X):
        xb = self.model_mar(X)
        return torch.exp(xb)
    def forward_z(self, Z):
        zv = self.model_con(Z)
        return torch.exp(zv)
    def forward(self, X, Z):
        return self.forward_x(X) * self.forward_z(Z)
    def predict(self, X, Z=None):
        with torch.inference_mode():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            if Z is not None: 
                Z = torch.tensor(Z, dtype=torch.float32, device=self.device)            
                pred = self.forward(X, Z).flatten()
            else:
                pred = self.forward_x(X).flatten()
            pred = torch.exp(pred) - 0.5
        pred = pred.to('cpu').numpy()        
        return np.where(pred<0, 1e-3, pred)
    # ----------------------------------------------------------
    def data_to_tensor_dict(self, y, X, Z):
        tensor_dict = {}
        tensor_dict['y'] = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        tensor_dict['X'] = torch.tensor(X, dtype=torch.float32, device=self.device)
        tensor_dict['Z'] = torch.tensor(Z, dtype=torch.float32, device=self.device)        
        return tensor_dict
    # ----------------------------------------------------------
    def get_optimizer(self, optimizer_name, learning_rate, **kwargs):
        try:
            optimizer_class = getattr(optim, optimizer_name)                
            optimizer = optimizer_class([
                {'params': self.model_mar.parameters(), 'lr': learning_rate},
                {'params': self.model_con.parameters(), 'lr': learning_rate},
                {'params': [self.log_sig2u], 'lr': learning_rate},
                {'params': [self.log_sig2e], 'lr': learning_rate}
            ], **kwargs)
            return optimizer
        except AttributeError:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")
    # ----------------------------------------------------------
    def neg_hlik(self, y, X, Z, epsilon=1e-8):
        m = self.forward(X, Z)
        neg_hlik = torch.mean((y-m)**2)/(torch.exp(self.log_sig2e)+epsilon)/2
        v = self.model_con.weight # (1, n_clusters)
        temp_nllv = torch.mean(v**2)/(torch.exp(self.log_sig2u)+epsilon)/2
        neg_hlik += temp_nllv*self.n_clusters/self.sample_size
        if self.log_sig2u.requires_grad:
            neg_hlik += self.log_sig2e/2
            canonicalizer = torch.mean(torch.log(1+self.cluster_sizes/(torch.exp(self.log_sig2e-self.log_sig2u)+epsilon)))/2
            neg_hlik += canonicalizer*self.n_clusters/self.sample_size
        return neg_hlik
    # ----------------------------------------------------------
    def neg_mlik(self, y, X, Z, epsilon=1e-8):
        errs = y - self.forward_x(X)
        ycov = torch.exp(self.log_sig2e)*torch.eye(y.shape[0]).to(self.device) + Z@Z.T*torch.exp(self.log_sig2u)        
        neg_mlik = (errs.T@torch.linalg.solve(ycov, errs) + torch.linalg.slogdet(ycov)[1])/y.shape[0]/2
        return neg_mlik
    # ----------------------------------------------------------
    def neg_clik(self, y, X, Z=None, epsilon=1e-8):
        if Z is None: m = self.forward_x(X)
        else: m = self.forward(X, Z)
        neg_clik = torch.mean((y-m)**2)/(torch.exp(self.log_sig2e)+epsilon)/2 + self.log_sig2e/2
        return neg_clik
    # ----------------------------------------------------------
    def adjust_random_effects(self):
        with torch.no_grad():
            v = self.model_con.weight.data
            adjust_term = torch.mean(v)
            self.model_con.weight.data -= adjust_term
            self.model_mar[-1].bias.data += adjust_term
    # ----------------------------------------------------------
    def initialize(self, train_data, valid_data, optimizer, learning_rate, sig2e_init=1., sig2u_init=1.):
        self.train_data  = self.data_to_tensor_dict(*train_data)
        self.valid_data  = self.data_to_tensor_dict(*valid_data)
        self.sample_size = self.train_data['y'].shape[0]
        # get number of total counts in each cluster
        self.cluster_sizes = torch.sum(self.train_data['Z'], axis=0).view(-1,1)
        self.y_sum_train = self.train_data['Z'].T @ self.train_data['y']
        self.y_sum_valid = self.valid_data['Z'].T @ self.valid_data['y']
        # log variance of random effects u
        self.log_sig2e = nn.Parameter(torch.log(torch.tensor(sig2e_init, dtype=torch.float32, device=self.device)))
        self.log_sig2u = nn.Parameter(torch.log(torch.tensor(sig2u_init, dtype=torch.float32, device=self.device)))
        # get optimizer and lr_scheduler
        self.optimizer = self.get_optimizer(optimizer, learning_rate)
        self.history = {key: [] for key in 
            ['time', 'train_loss', 'valid_loss', 'sig2u', 'sig2e']
        }
    # ----------------------------------------------------------
    def train_hlik(self, train_data, valid_data,
        optimizer, learning_rate, batch_size, max_epochs, patience,
        sig2e_init=1., sig2u_init=1., adjust=True, shuffle=True):
        # initialization -------------------------------------------
        self.loss_type = 'hlik'
        self.initialize(train_data, valid_data, optimizer, learning_rate, sig2e_init, sig2u_init)
        patience_count = 0
        min_valid_loss = float('inf')
        # training -------------------------------------------------
        dataset_train = TensorDataset(*self.train_data.values())
        self.start_time = time.time()
        for epoch in range(max_epochs):
            loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=False)
            temp_train_losses = []
            for y_batch, X_batch, Z_batch in loader:
                y_batch = y_batch.to(self.device)
                X_batch = X_batch.to(self.device)
                Z_batch = Z_batch.to(self.device)
                self.optimizer.zero_grad()
                loss = self.neg_hlik(y_batch, X_batch, Z_batch)
                loss.backward()
                self.optimizer.step()
                temp_train_losses.append(loss.item()*X_batch.size(0)/self.sample_size)
            with torch.no_grad():
                if adjust: self.adjust_random_effects()
            self.update_history(train_loss=np.mean(temp_train_losses))
            # early stopping -------------------------------------------
            if self.history['valid_loss'][-1] > min_valid_loss:
                patience_count += 1
            else:
                min_valid_loss = self.history['valid_loss'][-1]
                patience_count = 0
            if patience_count >= patience:
                break
    # ----------------------------------------------------------
    def train_mlik(self, train_data, valid_data,
        optimizer, learning_rate, batch_size, max_epochs, patience,
        sig2e_init=1., sig2u_init=1., adjust=False, shuffle=False):
        # initialization -------------------------------------------
        self.loss_type = 'hlik'
        self.initialize(train_data, valid_data, optimizer, learning_rate, sig2e_init, sig2u_init)
        patience_count = 0
        min_valid_loss = float('inf')
        # training -------------------------------------------------
        dataset_train = TensorDataset(*self.train_data.values())
        self.start_time = time.time()
        for epoch in range(max_epochs):
            loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=False)
            temp_train_losses = []
            for y_batch, X_batch, Z_batch in loader:
                y_batch = y_batch.to(self.device)
                X_batch = X_batch.to(self.device)
                Z_batch = Z_batch.to(self.device)
                self.optimizer.zero_grad()
                loss = self.neg_mlik(y_batch, X_batch, Z_batch)
                loss.backward()
                self.optimizer.step()
                temp_train_losses.append(loss.item()*X_batch.size(0)/self.sample_size)
            with torch.no_grad():
                fX_train = self.forward_x(self.train_data['X'])
                vpred = (self.train_data['Z'].T@(self.train_data['y']-fX_train))/(self.cluster_sizes+torch.exp(self.log_sig2e-self.log_sig2u))
                self.model_con.weight.data = vpred.view(1,-1)
            self.update_history(train_loss=np.mean(temp_train_losses))
            # early stopping -------------------------------------------
            if self.history['valid_loss'][-1] > min_valid_loss:
                patience_count += 1
            else:
                min_valid_loss = self.history['valid_loss'][-1]
                patience_count = 0
            if patience_count >= patience:
                break            
    # ----------------------------------------------------------
    def train_clik(self, train_data, valid_data, is_conditional,
        optimizer, learning_rate, batch_size, max_epochs, patience, sig2e_init=1., shuffle=True):
        # initialization -------------------------------------------
        self.loss_type = 'clik'
        self.initialize(train_data, valid_data, optimizer, learning_rate, sig2e_init)
        self.log_sig2u.requires_grad_(False)
        if not is_conditional: self.valid_data['Z'] = None
        patience_count = 0
        min_valid_loss = float('inf')
        # training -------------------------------------------------
        dataset_train = TensorDataset(*self.train_data.values())
        self.start_time = time.time()
        for epoch in range(max_epochs):
            loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=False)
            temp_train_losses = []
            for y_batch, X_batch, Z_batch in loader:
                y_batch = y_batch.to(self.device)
                X_batch = X_batch.to(self.device)
                Z_batch = Z_batch.to(self.device)
                self.optimizer.zero_grad()
                if not is_conditional: Z_batch = None
                loss = self.neg_clik(y_batch, X_batch, Z_batch)
                loss.backward()
                self.optimizer.step()
                temp_train_losses.append(loss.item()*X_batch.size(0)/self.sample_size)                        
            self.update_history(train_loss=np.mean(temp_train_losses))
            # early stopping
            if self.history['valid_loss'][-1] > min_valid_loss:
                patience_count += 1
            else:
                min_valid_loss = self.history['valid_loss'][-1]
                patience_count = 0
            if patience_count >= patience:
                break
    # ----------------------------------------------------------
    def update_history(self, train_loss):
        with torch.no_grad():
            if   self.loss_type == 'hlik': valid_loss = self.neg_hlik(*self.valid_data.values()).item()
            elif self.loss_type == 'clik': valid_loss = self.neg_clik(*self.valid_data.values()).item()
        self.history['time'].append(time.time()-self.start_time)
        self.history['train_loss'].append(train_loss)
        self.history['valid_loss'].append(valid_loss)
        self.history['sig2u'].append(torch.exp(self.log_sig2u).item())
        self.history['sig2e'].append(torch.exp(self.log_sig2e).item())
    # ----------------------------------------------------------