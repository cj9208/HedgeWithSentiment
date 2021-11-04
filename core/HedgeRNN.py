
import torch
import torch.nn as nn
import torch.optim as optim

import os 
import datetime 
import copy
import collections

import numpy as np 
import pandas as pd 

from tqdm import tqdm 


def loss_fn(delta_pred, dV, dS, invWeight=None, mode='mean', d=2): # mode : mean / sum 
    # delta_pred, dV, dS should be torch tensor/numpy array  of the same size
    t = dV - delta_pred * dS 
    if invWeight is not None:
        t = t / invWeight
    if isinstance(t, torch.Tensor):
        if mode == 'sum':
            return torch.sum(t**d)
        return torch.mean(t**d)
    if mode == 'sum':
        return np.sum(t**d)
    return np.mean(t**d)


class DataLoader:

    def __init__(self, df_feature, dV, dS, date, vix_return, device, seq_len=16,  batch_size=64, \
        shuffle=True, pin_memory=True, detail=False):

        assert len(df_feature) == len(dV) and len(df_feature) == len(dS)
        assert len(df_feature) == len(date) 
        self.detail = detail

        self.df_feature = df_feature.values
        self.dV = dV.values
        self.dS = dS.values

        self.date = pd.to_datetime(date)
        self.index = df_feature.iloc[:].index
        #print(self.index)

        self.vix_return = vix_return.values
        self.date_index = vix_return.index 

        self.seq_len = seq_len
        self.batch_size = batch_size

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.dV = torch.tensor(self.dV, dtype=torch.float, device=device)
            self.dS = torch.tensor(self.dS, dtype=torch.float, device=device)
            self.vix_return = torch.tensor(self.vix_return, dtype=torch.float, device=device)
  
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        # option_count contains the length of time series for each 
        self.option_count = dV.groupby(level=0).size().values  
        self.option_count_all = np.sum(self.option_count)

        self.option_index = np.roll(np.cumsum(self.option_count), 1)
        self.option_index[0] = 0

    @property
    def batch_length(self):
        return self.option_count_all // self.batch_size + 1

    def iter_batch(self):

        indices = np.arange(self.option_count_all)
        if self.shuffle:
            np.random.shuffle(indices)

        for di in range(len(indices))[::self.batch_size]:
            if self.option_count_all < self.batch_size + di :
                up = self.option_count_all
            else:
                up = di + self.batch_size

            indices_batch = indices[di:up]

            df_feature_batch = self.df_feature[indices_batch]
            dV_batch = self.dV[indices_batch]
            dS_batch = self.dS[indices_batch]

            price = []
            for t in indices_batch:
                dt = self.date.iloc[t]
                idx = self.date_index.get_loc(dt)
                price.append(self.vix_return[idx-self.seq_len+1:idx+1] )
            if self.pin_memory:
                price = torch.stack(price, axis=0)
            else :
                price = np.stack(price, axis=0)

            if self.detail:
                yield self.index[indices_batch], df_feature_batch, dV_batch, dS_batch, price
            else:      
                yield df_feature_batch, dV_batch, dS_batch, price


class GRU(nn.Module):

    def __init__(self, d_feat, d_feat_seq, hidden_size=32, num_layers=1, num_output=1, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat_seq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        #self.feature_extend = nn.Linear(d_feat,hidden_size)
        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2+hidden_size, num_output) #nn.Linear(2*hidden_size, num_output)
        )
        
        self.d_feat = d_feat
        self.d_feat_seq = d_feat_seq

    def forward(self, x):
        # [N, 2], [N, F, T]
        feature, x = x  
        x = x.reshape(len(x), self.d_feat_seq, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        out, _ = self.rnn(x)  
        out = out[:, -1, :]
        #feature = self.feature_extend(feature)
        out = torch.cat((out, feature), axis=1)
        return self.fc_out(out).squeeze()

class GRU2(nn.Module):

    def __init__(self, d_feat, d_feat_seq, hidden_size=32, num_layers=1, num_output=1, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat_seq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.feature_extend = nn.Linear(d_feat,hidden_size)
        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*hidden_size, num_output)
        )
        
        self.d_feat = d_feat
        self.d_feat_seq = d_feat_seq

    def forward(self, x):
        # [N, 2], [N, F, T]
        feature, x = x  
        x = x.reshape(len(x), self.d_feat_seq, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        out, _ = self.rnn(x)  
        out = out[:, -1, :]
        feature = self.feature_extend(feature)
        out = torch.cat((out, feature), axis=1)
        return self.fc_out(out).squeeze()

def get_model(model_name):
    if model_name.upper() == 'GRU':
        return GRU
    elif model_name.upper() == 'GRU2':
        return GRU2
    raise ValueError('unknown model name `%s`'%model_name)

def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params

def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)
    

def train_epoch(epoch, model, optimizer, train_loader, para, progressive_bar=False):
 
    model.train()
    if progressive_bar:
        iter_progressive = tqdm(train_loader.iter_batch(), total=train_loader.batch_length)
    else:
        iter_progressive = train_loader.iter_batch()

    for feature, dV, dS, price in iter_progressive:
        pred = model((feature, price))
        loss = loss_fn(pred, dV, dS)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), para['max_weight'])
        optimizer.step()
    
        torch.cuda.empty_cache() # docker memory leak

    return 

def test_epoch(epoch, model, test_loader, para, progressive_bar=False, prefix='Test'):
    model.eval()

    loss_all = []
    if progressive_bar:
        iter_progressive = tqdm(test_loader.iter_batch(), total=test_loader.batch_length)
    else:
        iter_progressive = test_loader.iter_batch()
    for feature, dV, dS, price in iter_progressive: 
        with torch.no_grad():
            pred = model((feature, price))
        loss = loss_fn(pred, dV, dS)
        loss_all.append(loss.item())
        torch.cuda.empty_cache() # docker memory leak
   
    loss = np.mean(loss_all)

    return loss

def infer(model, test_loader, para, progressive_bar=False, prefix='Infer'):
    model.eval()

    delta_all = []
    if progressive_bar:
        iter_progressive = tqdm(test_loader.iter_batch(), total=test_loader.batch_length)
    else:
        iter_progressive = test_loader.iter_batch()

    for idx, feature, dV, dS, price in iter_progressive: 
        with torch.no_grad():
            pred = model((feature, price))

        t = pred.cpu().detach().numpy() 
        t = pd.Series(t, index=idx)
       
        delta_all.append(t)
    
    #print(pd.concat(delta_all))
    return pd.concat(delta_all) 



def create_loaders(df, vix_return, para, train=True, detail=False):

    #pprint('..load data')
    # shuffle
    #df = df.sample(frac=1).reset_index(drop=True)

    df_feature = df[['normalized_T', 'delta'] ]
    dV = df['dV']
    dS = df['dS']
    date = df['date']

    #pprint('..build loader')
    if train:
        N = df.shape[0]
        valid_idx = int(N*para['train_percent'])

        slc = range(valid_idx)
        train_loader = DataLoader(df_feature.iloc[slc], dV.iloc[slc], dS.iloc[slc], date.iloc[slc], vix_return,
                                device=para['device'], seq_len=para['seq_len'],  batch_size=para['batch_size'],
                                shuffle=para['shuffle'], pin_memory=para['pin_memory'])

        slc = range(valid_idx, N)
        valid_loader = DataLoader(df_feature.iloc[slc], dV.iloc[slc], dS.iloc[slc], date.iloc[slc], vix_return,
                                device=para['device'], seq_len=para['seq_len'],  batch_size=para['batch_size'],
                                shuffle=para['shuffle'], pin_memory=para['pin_memory'])

        return train_loader, valid_loader
    else:
        data_loader = DataLoader(df_feature, dV, dS, date, vix_return,
                                device=para['device'], seq_len=para['seq_len'],  batch_size=para['batch_size'],
                                shuffle=para['shuffle'], pin_memory=para['pin_memory'],
                                detail=detail)
        return data_loader



para_default = {
    'data_path':'../data_SPX/',
    'model_name':'GRU',
    'd_feat': 2, 
    'd_feat_seq':2,
    'hidden_size':16,
    'num_layers':1,
    'dropout':0.0,
    'pin_memory': True,
    'shuffle': True,
    'train_percent':0.8,
    'seq_len':22,  
    'batch_size':1024, 
    'lr': 5e-4,
    'n_epochs':100,
    'early_stop':10,
    'smooth_steps':5,
    'clip_weight': True,
    'max_weight':3.0,
    'overwrite':True,
    'continue_train':False
}


class HedgeRNN():
    def __init__(self, para=para_default):   
        self.para = para
      
        self.set_model()
        # continue training or start new tratining
        self.continue_train = para['continue_train']

    def set_model(self):
        para = self.para

        # define model 
        model_name = para['model_name']
        d_feat = para['d_feat']
        d_feat_seq = para['d_feat_seq']
        hidden_size = para['hidden_size']
        num_layers = para['num_layers']
        dropout = para['dropout']

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = get_model(model_name)(d_feat, d_feat_seq, hidden_size, num_layers, 1, dropout)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=para['lr'])

        self.para['device'] = device
        self.model = model
        self.optimizer = optimizer

    def fit(self, df, dV, dS, vix_return=None, progressive_bar=False):
        df = df.copy()
        para = self.para
        if self.continue_train is False:
            self.set_model()
        model = self.model
        optimizer = self.optimizer

        # read sequential data
        if vix_return is not None:
            vix_return = pd.read_hdf(self.para['data_path'] + 'vix_return.h5')
        # vix	log_return
        if self.para['TYPE'] == 'C':
            vix_return = vix_return[['vix']]
        else:
            vix_return = vix_return[['log_return']]
        
        #print(vix_return.shape)

        self.vix_return = vix_return

        if progressive_bar:
            pprint('create model...')
        train_loader, valid_loader = create_loaders(df, self.vix_return, para)

        suffix = "risk_model_%s_hidden_size_%s_number_layers_%s_lr%s_dropout%s"%(
            para['model_name'].lower(), para['hidden_size'], para['num_layers'], para['lr'], para['dropout'] )
        output_path = '../output/' + suffix
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        best_score = {}
        best_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())
        params_list = collections.deque(maxlen=para['smooth_steps'])
        for epoch in range(self.para['n_epochs']):
            print('Epoch : {}, time : {}'.format(epoch, datetime.datetime.now().strftime("%H:%M:%S")))
            if progressive_bar:
                pprint('Epoch:', epoch)
                pprint('training...')
            train_epoch(epoch, model, optimizer, train_loader, para, progressive_bar)

            torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
            torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

            params_ckpt = copy.deepcopy(model.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            model.load_state_dict(avg_params)
            if progressive_bar:
                pprint('evaluating...')
            train_score = test_epoch(epoch, model, train_loader, para, progressive_bar, prefix='Train')
            val_score = test_epoch(epoch, model, valid_loader, para, progressive_bar, prefix='Valid')
            if progressive_bar:
                pprint('Epoch %d train %.6f, valid %.6f'%(epoch, train_score, val_score))

            model.load_state_dict(params_ckpt)

            if val_score < best_score.get('valid', np.inf):
                best_score = {'train': train_score, 'valid': val_score}
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(avg_params)
            else:
                stop_round += 1
                if stop_round >= self.para['early_stop']:
                    pprint('early stop')
                    break
            
            print(' '*4 + 'Train error : {}, Validation error : {}'.format(train_score, val_score) )

        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        self.model = model
        return best_epoch, best_score

    def predict(self, df, dV=None, dS=None):
        df = df.copy()
        data_loader = create_loaders(df, self.vix_return, self.para, train=False, detail=True)
        hedge_ratio = infer(self.model, data_loader, self.para, prefix='Infer')
        return hedge_ratio
        
    def score(self, df, dV, dS, printout=False):
        delta_pred = self.predict(df)
        loss = loss_fn(delta_pred, dV, dS)
        if printout:
            pprint('Loss : %s'%(loss))
        return loss





  