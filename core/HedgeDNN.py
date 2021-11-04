
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

    def __init__(self, df_feature, dV, dS, device, batch_size=64, shuffle=True, pin_memory=True, detail=False):

        assert len(df_feature) == len(dV) and len(df_feature) == len(dS)
        self.detail = detail

        self.df_feature = df_feature.values
        self.dV = dV.values
        self.dS = dS.values

        self.batch_size = batch_size

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.dV = torch.tensor(self.dV, dtype=torch.float, device=device)
            self.dS = torch.tensor(self.dS, dtype=torch.float, device=device)
       
        self.index = df_feature.index

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
            if self.detail:
                yield self.index[indices_batch], df_feature_batch, dV_batch, dS_batch
            else:      
                yield df_feature_batch, dV_batch, dS_batch

class DNN(nn.Module):
    def __init__(self, d_feat, hidden_size=64, num_layers=2, num_output=1, dropout=0.0, TYPE='C'):
        super().__init__()

        self.TYPE = TYPE

        network = []
        input_size = d_feat
        network.append(nn.BatchNorm1d(num_features=d_feat))
        for i in range(num_layers):
            #network.append(nn.BatchNorm1d(input_size))
            if i > 0:
                network.append(nn.BatchNorm1d(num_features=hidden_size))
                network.append(nn.ReLU())
            #network.append(nn.Dropout(dropout))
            network.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        network.append(nn.ReLU())
        #network.append(nn.Sigmoid())
        self.network=nn.Sequential(*network)

        #output = [
        #    nn.Linear(input_size, num_output),
        #    nn.ReLU() ]
        #self.fc = nn.Sequential(*output)
        self.fc = nn.Linear(input_size, num_output)

    def forward(self, x):
        x = self.network(x)
        out = self.fc(x)

        out = torch.sigmoid(out) # for call option, should be bigger than zero
        if self.TYPE == 'P':  
            out = -out

        #if self.TYPE == 'C':  # for call option, should be bigger than zero
        #    out = nn.functional.relu(out)  
        #else:   # for put option, should be less than zero
        #    out = -nn.functional.relu(-out)  
        return out.squeeze()



def get_model(model_name, TYPE='C'):
    if model_name.upper() == 'DNN':
        return DNN

    else :
        print('Wrong type')
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
    for feature, dV, dS in iter_progressive:
        pred = model(feature)
        loss = loss_fn(pred, dV, dS, d=para['loss_d'])
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
    for feature, dV, dS in iter_progressive:
        with torch.no_grad():
            pred = model(feature)
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

    for idx, feature, dV, dS in iter_progressive:
        with torch.no_grad():
            pred = model(feature)

        t = pred.cpu().detach().numpy() 
        t = pd.Series(t, index=idx)
        delta_all.append(t)
 
    return pd.concat(delta_all)


def create_loaders(df,  para, stock_price=None,sentiment=False, train=True, detail=False):

    #pprint('..load data')
    # shuffle
    #df = df.sample(frac=1).reset_index(drop=True)

    if para['feature_set'] == 1:
        feature_list = ['log_forward_moneyness', 'normalized_T', 'impl_volatility'] 
    elif para['feature_set'] == 2:
        feature_list = ['moneyness', 'normalized_T', 'delta'] 
    elif para['feature_set'] == 22:
        feature_list = ['normalized_T', 'delta']  # delta is similar to moneyness? 
    elif para['feature_set'] == 23:
        feature_list = ['moneyness', 'normalized_T', 'delta', 'log_return'] 
    elif para['feature_set'] == 24:
        feature_list = ['normalized_T', 'delta', 'log_return'] 
    elif para['feature_set'] == 242:
        feature_list = ['normalized_T', 'delta', 'SPX_log_return'] 
    elif para['feature_set'] == 25:
        feature_list = ['normalized_T', 'delta', 'vix'] 
    elif para['feature_set'] == 26:
        feature_list = ['normalized_T', 'delta', 'impl_volatility'] #'vol_22' 
    elif para['feature_set'] == 27:
        feature_list = ['normalized_T', 'delta', 'vix', 'impl_volatility'] 

    elif para['feature_set'] == 3:
        feature_list = [ 'log_return', 'delta', 'normalized_T', 'vix'] 
    elif para['feature_set'] == 31:
        feature_list = ['moneyness', 'delta', 'normalized_T', 'vix'] 
    elif para['feature_set'] == 32:
        feature_list = ['moneyness', 'normalized_T', 'delta', 'log_return', 'vix'] 
    elif para['feature_set'] == 33:
        feature_list = ['normalized_T', 'delta'] 
        add_list = ['log_return', 'vix']
        tmp = []
        for suffix in ['week', 'month']:
            for name in add_list:
                tmp.append(name+'_'+suffix)
        add_list.extend(tmp)
        feature_list.extend(add_list)

    else:
        print('Wrong feature set')

    # scale 
    if para.get('scale', False):
        for name in feature_list:
            if 'log_return' in name or 'vix' in name or 'normalized_T' in name:
                df[name] = df[name]*100


    if para['vega'] is True:
        feature_list.append('normalized_vega_log')

    if para['sentiment'] is True:
        feature_list.extend(['day_CSS'])
    if para['extreme_sentiment'] is True:
        feature_list.extend(['max_CSS', 'min_CSS'])
    if para['high_sentiment'] is True:
        feature_list.extend(['high_CSS', 'low_CSS'])

    if para['vix'] is True:
        feature_list.extend(['log_return', 'vol_22', 'vix'])

    if para['shuffle']:
        indices = np.arange(df.shape[0])
        np.random.shuffle(indices)
        df = df.iloc[indices]

    df_feature = df[feature_list]
    
    dV = df['dV']
    dS = df['dS']

    
    #pprint('..build loader')
    if train:
        N = df.shape[0]
        valid_idx = int(N*para['train_percent'])

        slc = range(valid_idx)
        train_loader = DataLoader(df_feature.iloc[slc], dV.iloc[slc], dS.iloc[slc],
                                device=para['device'],  batch_size=para['batch_size'],
                                shuffle=para['shuffle'], pin_memory=para['pin_memory'])

        slc = range(valid_idx, N)
        valid_loader = DataLoader(df_feature.iloc[slc], dV.iloc[slc], dS.iloc[slc], 
                                device=para['device'],  batch_size=para['batch_size'],
                                shuffle=para['shuffle'], pin_memory=para['pin_memory'])

        return train_loader, valid_loader
    else:
        data_loader = DataLoader(df_feature, dV, dS,
                                device=para['device'], batch_size=para['batch_size'],
                                shuffle=para['shuffle'], pin_memory=para['pin_memory'], 
                                detail=detail)
        return data_loader



para_default = {
    'data_path':'../data/',
    'model_name':'GRU',
    'd_feat': 3, 
    'hidden_size':64,
    'num_layers':2,
    'dropout':0.0,
    'pin_memory': True,
    'shuffle': True,
    'train_percent':0.8,
    'batch_size':64, 
    'lr': 5e-4,
    'n_epochs':100,
    'early_stop':20,
    'smooth_steps':5,
    'clip_weight': True,
    'max_weight':3.0,
    'overwrite':True,
    'continue_train':False,
    'loss_d':2,
    'sentiment':False
}


class HedgeDNN():
    def __init__(self, para=para_default):   
        self.para = para

        if self.para['sentiment'] is True:
            # ESS/CSS + mean/ratio
            self.para['d_feat'] += 1

        if self.para['extreme_sentiment'] or self.para['high_sentiment']:
            # ESS/CSS + mean/ratio
            self.para['d_feat'] += 2

        if self.para['vix']:
            self.para['d_feat'] += 3

        if para['vega'] is True:
            self.para['d_feat'] += 1

            
        self.set_model()
        # continue training or start new tratining
        self.continue_train = para['continue_train']



    def set_model(self):
        para = self.para

        # define model 
        model_name = para['model_name']
        d_feat = para['d_feat']
        hidden_size = para['hidden_size']
        num_layers = para['num_layers']
        dropout = para['dropout']
        TYPE=para['TYPE']

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = get_model(model_name, TYPE=TYPE)(d_feat, hidden_size, num_layers, 1, dropout, TYPE=TYPE)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=para['lr'])

        self.para['device'] = device
        self.model = model
        self.optimizer = optimizer



    def fit(self, df, dV, dS, progressive_bar=False):
        
        df = df.copy()
        
        para = self.para
        if self.continue_train is False:
            self.set_model()
        model = self.model
        optimizer = self.optimizer

        if progressive_bar:
            pprint('create model...')
        
        if self.para['vega'] is True:
            df['vega_log'] = df['vega'].apply(np.log)
            self.vega_mean = df['vega_log'].mean()
            self.vega_std = df['vega_log'].std()
            df['normalized_vega_log'] = (df['vega_log'] - self.vega_mean) / self.vega_std

        train_loader, valid_loader = create_loaders(df, para)

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

        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)

        self.model = model

        return best_epoch, best_score

  
    def predict(self, df, dV=None, dS=None):
        df = df.copy()
        data_loader = create_loaders(df, self.para, train=False, detail=True)
        hedge_ratio = infer(self.model, data_loader, self.para, prefix='Infer')
        return hedge_ratio
        
    def score(self, df, dV, dS, printout=False):
        delta_pred = self.predict(df)
        loss = loss_fn(delta_pred, dV, dS)
        if printout:
            pprint('Loss : %s'%(loss))
        return loss





    
