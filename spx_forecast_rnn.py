
import pandas as pd 
import numpy as np 
import time
import os 
from core.HedgeRNN import HedgeRNN
import argparse


def loss_fn(delta_pred, dV, dS): 
    # delta_pred, dV, dS should be torch tensor/numpy array  of the same size
    t = dV - delta_pred * dS 
    return np.mean(t**2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SPX', action='store_false') #  default true 
    parser.add_argument('--cp_flag', type=str, default = 'C')
    parser.add_argument('--shuffle', action='store_true') 
    parser.add_argument('--trade', action='store_false') 
    parser.add_argument('--suffix', type=str, default='') 
    parser.add_argument('--model_name', type=str, default='GRU')
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--n_epoch', type=int, default=100)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(args)

    # read data
    if args.SPX:
        data_path = '../data_SPX/SPX_Option_'+args.cp_flag+'_E.h5'
    else:
        data_path = '../data_All/All_Option_'+args.cp_flag+'_A.h5'
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

    # add to existing file
    suffix = 'hidden_size_'+str(args.hidden_size)+'_num_layer_'+str(args.num_layer)+'_'+args.suffix
    filename = '../data_SPX/SPX_Option_'+args.cp_flag+'_E_activation_'+suffix+'.h5'
    if os.path.exists(filename):
        print('Test file already exists')
        df_test = pd.read_hdf(filename)

    #print(df_test)
    print('Number of training samples : ', df_train.shape[0])
    print('Number of test samples     : ', df_test.shape[0])

    vix_ret = pd.read_hdf('../data_SPX/vix_return.h5')

    # normalized_T, delta 
    para_RNN = {
        'data_path':'../data_SPX/',
        'model_name':args.model_name, # or GRU2 with one layer of FCNN after BS delta, TTM 
        'd_feat': 2, 
        'd_feat_seq':1,
        'hidden_size':args.hidden_size,
        'num_layers':args.num_layer,
        'dropout':0.0,
        'pin_memory': True,
        'shuffle': True,
        'train_percent':0.8,
        'seq_len':22,  
        'batch_size':1024, 
        'lr': 5e-4,
        'n_epochs':args.n_epoch,
        'early_stop':10,
        'smooth_steps':5,
        'clip_weight': True,
        'max_weight':3.0,
        'TYPE': args.cp_flag,
        'overwrite':True,
        'continue_train':False

    }


    models = {
        'RNN': HedgeRNN(para=para_RNN)
    }
    train_loss = {}
    test_loss = {}

    
    print('delta')
    print('train loss:  ', loss_fn(df_train['delta'], df_train['dV'], df_train['dS']) )
    delta_loss = loss_fn(df_test['delta'], df_test['dV'], df_test['dS'])
    print('test loss :  ',  delta_loss)

    for name, model in models.items():
        start = time.time()
        print(name)
        print('Training')
        model.fit(df_train, df_train['dV'], df_train['dS'], vix_return=vix_ret, progressive_bar=False)
        train_loss[name] = model.score(df_train, df_train['dV'], df_train['dS'])

        print('Testing')
        #print(df_test.index)
        df_test[name] = model.predict(df_test, df_test['dV'], df_test['dS'])
        test_loss[name] = loss_fn(df_test[name], df_test['dV'], df_test['dS']) 
        
        #print(df_test)
        print('train loss:  ', train_loss[name])
        print('test loss :  ', test_loss[name])
        print('gain      :  ', 1-test_loss[name]/delta_loss)
        print('Method {} takes {} seconds'.format(name, time.time()-start))

        print('='*20)

    losses = pd.concat({
        'train': pd.Series(train_loss),
        'test': pd.Series(test_loss)
    }, axis=1)

    print(losses)

    df_test.to_hdf(filename, key='SPX_Option')
