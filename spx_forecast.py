
import pandas as pd 
import numpy as np 
import time
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
    parser.add_argument('--SPX', action='store_false') #  default true 
    parser.add_argument('--cp_flag', type=str, default = 'C')
    parser.add_argument('--shuffle', action='store_true') 
    parser.add_argument('--trade', action='store_false') 
    parser.add_argument('--suffix', type=str, default='') 

    parser.add_argument('--hidden_size', type=int, default=128)
    
    parser.add_argument('--plot', action='store_true')
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

    if args.plot:
        df_test = pd.read_hdf('../data_SPX/SPX_Option_'+args.cp_flag+'_E_plot.h5')

    print('Number of training samples : ', df_train.shape[0])
    print('Number of test samples     : ', df_test.shape[0])

    
    

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
            'feature_set': 24,
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

    #moneyness, normalized_T, delta, log_return 
    para_DNN11 = {    
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
            'feature_set': 23,
            'scale': True
    }

    #moneyness, normalized_T, delta, vix
    para_DNN101 = {    
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
            'feature_set': 31,
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


    # add week, month feature (average over a week/month)
    para_DNN3 = {    
            'model_name':'DNN',
            'd_feat': 8, 
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
            'TYPE': 'C',
            'feature_set': 33,
            'scale': True
    }

    para_MVDNN = {
        'model_name':'DNN',
        'd_feat': 4, 
        'hidden_size':128,
        'num_layers':3,
        'dropout':0.0,
        'pin_memory': True,
        'shuffle': True,
        'train_percent':0.8,
        'batch_size':1024, 
        'lr': 5e-4,
        'n_epochs':100,
        'early_stop':10,
        'smooth_steps':5,
        'clip_weight': True,
        'max_weight':3.0,
        'TYPE':'C',
        'continue_train': False,
        'scale':100
    }

    models = {
        #'MVdelta': MVdelta(), 
        'DNN0': HedgeDNN(para=para_DNN0),
        #'DNN1': HedgeDNN(para=para_DNN1),
        'DNN01': HedgeDNN(para=para_DNN01),
        'DNN001': HedgeDNN(para=para_DNN001),
        #'DNN11': HedgeDNN(para=para_DNN11),
        #'DNN101': HedgeDNN(para=para_DNN101),
        #'DNN011': HedgeDNN(para=para_DNN011), 
        #'DNN3': HedgeDNN(para=para_DNN3),
        #'MVDNN': MVDNN_imp(para=para_MVDNN),  # how to transform implied volatility to hedge ratio
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
        model.fit(df_train, df_train['dV'], df_train['dS'], progressive_bar=False)
        train_loss[name] = model.score(df_train, df_train['dV'], df_train['dS'])

        #if name == 'MVdelta':
        #    print(model.reg.coef_)

        test_loss[name] = model.score(df_test, df_test['dV'], df_test['dS']) 
        df_test[name] = model.predict(df_test, df_test['dV'], df_test['dS'])
        #if 'MV' in name:
        #    df_test['res' + name] = df_test['dV'] - df_test['dS']*df_test['delta'] - df_test[name]
        #    test_loss[name] = np.mean(df_test['res' + name]**2)

        #else:
        test_loss[name] = loss_fn(df_test[name], df_test['dV'], df_test['dS']) 


        print('train loss:  ', train_loss[name])
        print('test loss :  ', test_loss[name])
        delta_loss = 0.1 if delta_loss < 0.1 else delta_loss
        print('gain      :  ', 1-test_loss[name]/delta_loss)
        print('Method {} takes {} seconds'.format(name, time.time()-start))

        print('='*20)

    losses = pd.concat({
        'train': pd.Series(train_loss),
        'test': pd.Series(test_loss)
    }, axis=1)

    print(losses*1e4)
    suffix = 'shuffle_'+ str(args.shuffle) + '_trade_'+ str(args.trade)+'_activation_'+args.suffix
    if args.plot:
        suffix = 'plot_' + suffix
    df_test.to_hdf('../data_SPX/SPX_Option_'+args.cp_flag+'_E_'+suffix+'.h5', key='SPX_Option')

    #cal_gain(df_test)

    #k2 = df_test
    #print('mean square of dimp', np.mean(k2['dimp']**2))
    #print('mean square error of MVdelta', np.mean((k2['dimp']-k2['MVdelta'])**2))
    #print('mean square error of MVDNN', np.mean((k2['dimp']-k2['MVDNN'])**2))
    
    

