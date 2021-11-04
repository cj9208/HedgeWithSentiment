
from dask.utils import filetext
import pandas as pd 
import numpy as np 
import time
import os

from torch import det 
from core.MVDNN_imp import MVDNN_imp 
from core.mvdelta import MVdelta
from core.HedgeDNN import HedgeDNN
import argparse

def cal_gain(k2):
    k2['detladS'] = k2['delta'] * k2['dS']
    for name in ['dimp', 'MVdelta', 'MVDNN']:
        k2[name+'vega'] = k2[name] / k2['stock_price'] * 1000 * k2['vega'] # scale 

    base = k2['dV'] - k2['detladS']
    base_score = np.mean(base**2)
    print('delta : {}'.format(base_score) )
    for name in ['dimp', 'MVdelta', 'MVDNN']: 
        tmp = base - k2[name+'vega']
        tmp_score = np.mean(tmp**2)
        tmp_gain = 1 - tmp_score / base_score 
        print('{:8}, score : {:.4f}, gain {:.4f}'.format(name, tmp_score, tmp_gain) )

def loss_fn(delta_pred, dV, dS): 
    # delta_pred, dV, dS should be torch tensor/numpy array  of the same size
    t = dV - delta_pred * dS 
    return np.mean(t**2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default = '../data_semiconduct/')
    #parser.add_argument('--data_path', type=str, default = '../data_All/')
    parser.add_argument('--cp_flag', type=str, default = 'C')
    parser.add_argument('--shuffle', action='store_true') 
    parser.add_argument('--trade', action='store_false') 
    parser.add_argument('--suffix', type=str, default='') 
    parser.add_argument('--transfer', action='store_true') # default false
    parser.add_argument('--stock_num', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128)
    
    args = parser.parse_args()
    return args


def test_ind(df_train, df_test, model, detail=False):
    mvoutput=[]
    mvloss = {}

    for ticker in df_train['ticker'].unique():

        train = df_train[df_train['ticker']==ticker]
        model.fit(train, train['dV'], train['dS'], progressive_bar=False)

        test = df_test[df_test['ticker']==ticker]
        mvoutput.append( model.predict(test, test['dV'], test['dS']) )

        mvloss[ticker] = np.mean((test['dV']-test['dS']*mvoutput[-1])**2)
        print(mvloss[ticker])

    hedge_ratio = pd.concat(mvoutput)

    if detail:
        return hedge_ratio, pd.Series(mvloss),

    return hedge_ratio 



if __name__ == '__main__':

    args = parse_args()
    print(args)
    suffix = args.suffix
    # read data
    data_path = args.data_path + 'All_Option_'+args.cp_flag+'_A.h5'
    df = pd.read_hdf(data_path)

    if args.trade :
        df = df[df['volume']>0]
       
    # setting
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
        N = int(df.shape[0]*0.9)
        df_train = df.iloc[:N]
        df_test = df.iloc[N:].copy()
    else:
        date_split = pd.Timestamp('20190101') 
        # Data of the last year is used for testing (easy to compare with previous methods)
        df_train = df[df['date'] < date_split]
        df_test = df[df['date'] >= date_split].copy()

    
    # Nian's feature set  moneyness, normalized_T, delta 
    para_DNN1 = {    
            'model_name':'DNN',
            'd_feat': 3, 
            'hidden_size':args.hidden_size,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 2,
            'scale': True
    }

    #moneyness, normalized_T, delta, log_return 
    para_DNN11 = {    
            'model_name':'DNN',
            'd_feat': 4, 
            'hidden_size':128,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 23,
            'scale': True
    }

    #moneyness, normalized_T, delta, vix
    para_DNN101 = {    
            'model_name':'DNN',
            'd_feat': 4, 
            'hidden_size':128,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 31,
            'scale': True
    }

    # add moneyness 
    para_DNN111 = {    
            'model_name':'DNN',
            'd_feat': 5, 
            'hidden_size':128,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 32,
            'scale': True
    }


    # normalized_T, delta 
    para_DNN0 = {    
            'model_name':'DNN',
            'd_feat': 2, 
            'hidden_size':args.hidden_size,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 22,
            'scale': True
    }

    # normalized_T, delta, spx_log_return 
    para_DNN01 = {    
            'model_name':'DNN',
            'd_feat': 3, 
            'hidden_size':args.hidden_size,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 242,
            'scale': True
    }

    # normalized_T, delta, vix
    para_DNN001 = {    
            'model_name':'DNN',
            'd_feat': 3, 
            'hidden_size':args.hidden_size,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 25,
            'scale': True
    } 

    # normalized_T, delta, implied_vol
    para_DNN0001 = {    
            'model_name':'DNN',
            'd_feat': 3, 
            'hidden_size':args.hidden_size,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 26,
            'scale': True
    } 

    # normalized_T, delta, vix + implied_vol
    para_DNN0011 = {    
            'model_name':'DNN',
            'd_feat': 4, 
            'hidden_size':args.hidden_size,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE':  args.cp_flag,
            'feature_set': 27,
            'scale': True
    } 

    # normalized_T, delta, log_return
    para_DNN0002 = {    
            'model_name':'DNN',
            'd_feat': 3, 
            'hidden_size':args.hidden_size,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 24,
            'scale': True
    } 
    
    # Hull's feature set, normalized_T, delta, log_return, vix 
    para_DNN011 = {    
            'model_name':'DNN',
            'd_feat': 4, 
            'hidden_size':args.hidden_size,
            'num_layers': 3,
            'dropout': 0.0,
            'pin_memory': True,
            'shuffle': True,
            'train_percent':0.8,
            'batch_size': 1024, 
            'lr': 5e-4,
            'n_epochs': 200,
            'early_stop': 10,
            'smooth_steps': 5,
            'clip_weight': True, 
            'max_weight': 3,  
            'continue_train': False,
            'loss_d': 2,
            'sentiment': False,
            'extreme_sentiment': False,
            'high_sentiment': False,
            'vix': False,
            'vega':False, 
            'TYPE': args.cp_flag,
            'feature_set': 3,
            'scale': True
    }


    train_loss = {}
    test_loss = {}
    train_loss['delta'] = loss_fn(df_train['delta'], df_train['dV'], df_train['dS'])
    test_loss['delta'] = delta_loss = loss_fn(df_test['delta'], df_test['dV'], df_test['dS'])
    
    print('delta')
    print('train loss:  ',  train_loss['delta'])
    print('test loss :  ',  test_loss['delta'])

    # individual model for each stock
    models_ind = {
        'MVdelta_ind': MVdelta(),
        'DNN0_ind': HedgeDNN(para=para_DNN0),
        'DNN01_ind': HedgeDNN(para=para_DNN01),
        'DNN001_ind': HedgeDNN(para=para_DNN001)
    }
    for name, model in models_ind.items():
        print(name)
        start = time.time()
        df_test[name] = test_ind(df_train, df_test, model, detail=False)
        test_loss[name] = np.mean((df_test['dV']-df_test['dS']*df_test[name])**2)

        print('test loss :  ', test_loss[name])
        print('gain      :  ', 1-test_loss[name]/delta_loss)
        print('Method {} takes {} seconds'.format(name, time.time()-start))



    if args.transfer:
        cnt = df.groupby('ticker')['dS'].count().sort_values(ascending=False)
        remain_stock_list = list(cnt.iloc[:args.stock_num].index) 
        df_train = df_train[df_train['ticker'].apply(lambda x: x in remain_stock_list)]

        suffix += '_' + str(args.stock_num)

    print('Number of training samples : ', df_train.shape[0])
    print('Number of test samples     : ', df_test.shape[0])

    # unified DNN model for all stocks 
    para_DNN0['hidden_size'] = 128
    para_DNN01['hidden_size'] = 128
    para_DNN001['hidden_size'] = 128
    models = {
        'MVdelta_all': MVdelta(),
        #'DNN1': HedgeDNN(para=para_DNN1),
        'DNN0': HedgeDNN(para=para_DNN0),
        'DNN01': HedgeDNN(para=para_DNN01),
        'DNN001': HedgeDNN(para=para_DNN001),
        #'DNN0011': HedgeDNN(para=para_DNN0011),
        #'DNN0001': HedgeDNN(para=para_DNN0001),
        #'DNN0002': HedgeDNN(para=para_DNN0002),
        #'DNN011': HedgeDNN(para=para_DNN011)
    }
    
    for name, model in models.items():
        start = time.time()
        print(name)
        model.fit(df_train, df_train['dV'], df_train['dS'], progressive_bar=False)
        train_loss[name] = model.score(df_train, df_train['dV'], df_train['dS'])
        #test_loss[name] = model.score(df_test, df_test['dV'], df_test['dS'])  
        df_test[name] = model.predict(df_test, df_test['dV'], df_test['dS'])

        test_loss[name] = loss_fn(df_test[name], df_test['dV'], df_test['dS']) 

        print('train loss:  ', train_loss[name])
        print('test loss :  ', test_loss[name])
        print('gain      :  ', 1-test_loss[name]/delta_loss)
        print('Method {} takes {} seconds'.format(name, time.time()-start))

        print('='*20)

    losses = pd.concat({
        'train': pd.Series(train_loss),
        'test': pd.Series(test_loss)
    }, axis=1)
    print(args)
    print(losses)

    suffix = 'trade_'+ str(args.trade)+'_transfer_'+str(args.transfer)+\
        '_hidden_size_'+str(args.hidden_size)+suffix
    file = args.data_path+'All_Option_'+str(args.cp_flag)+'_A_'+suffix+'.h5'
    #if os.path.exists(file):
    #    df_test = pd.read_hdf(file)
    df_test.to_hdf(file, key='All_Option')


    
    

