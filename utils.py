import numpy as np
import pandas as pd 



def cal_method_bucket_loss(df, name):
    t = df.copy()
    t['delta_bucket'] = t['delta'].apply(round_tenth)

    if 'MVdelta' in name:
        name_diff = name 
    else:
        name_diff = name + '_diff'
        t[name_diff] = t['dV'] - t[name]*t['dS']
    loss = t.groupby(['ticker', 'delta_bucket'])[name_diff].apply(lambda x: np.mean(x**2)).unstack()
    loss['overall'] = t.groupby('ticker')[name_diff].apply(lambda x: np.mean(x**2))
    return loss 

def cal_method_bucket_gain(df, N=25):
    name_list = df.columns[N:].to_list()

    loss0 = cal_method_bucket_loss(df, 'delta')
    gain_all = {}
    for name in name_list:
        loss = cal_method_bucket_loss(df, name)
        gain = 1 - loss/loss0 
        gain_all[name] = gain 
    return gain_all


def round_tenth(x, TYPE='C'):
    # 0.05 <= x <= 0.95 

    for i in range(1,10):
        if x >=-0.05 + i/10 and x < 0.05 + i/10:
            return i 
    return 9
 
def round_tenth_put(x):
    for i in range(9,0,-1):
        if x >=-0.05 - i/10 and x < 0.05 - i/10:
            return i 
    return 1

def cal_bucket_hedge_error_all(df, name='delta', residual=False, TYPE='C'):
    k = df
    k = k[k['month']>36].copy()
    if TYPE == 'C':
        k['delta_bucket'] = k['delta'].apply(round_tenth)
    elif TYPE == 'P':
        k['delta_bucket'] = k['delta'].apply(round_tenth_put)

    if not residual:
        k['hedge_error'] = k['dV'] - k[name]*k['dS']
        k['hedge_error_sq'] = k['hedge_error']**2 
    else: 
        k['hedge_error_sq'] = k[name]**2

    overall = k['hedge_error_sq'].mean()
    bucket = k.groupby('delta_bucket')['hedge_error_sq'].mean()
    
    return overall, bucket

def cal_bucket_gain_all(k, name, residual, subzero=False, TYPE='C', mean=True):
    overall, bucket = cal_bucket_hedge_error_all(k, 'delta', False, TYPE=TYPE)
    toverall, tbucket = cal_bucket_hedge_error_all(k, name, residual, TYPE=TYPE)

    gainoveral = 1 - toverall / overall
    gainbucket = 1 - tbucket / bucket

    if subzero:
        if TYPE == 'C':
            k['delta_bucket'] = k['delta'].apply(round_tenth)
        elif TYPE == 'P':
            k['delta_bucket'] = k['delta'].apply(round_tenth_put)

        for d in k['delta_bucket'].unique():
            if gainbucket[d] < 0:
                k.loc[k['delta_bucket']==d, name] = k.loc[k['delta_bucket']==d, 'delta'] 

        overall, bucket = cal_bucket_hedge_error_all(k, 'delta', False, TYPE=TYPE)
        toverall, tbucket = cal_bucket_hedge_error_all(k, name, residual, TYPE=TYPE)

        gainoveral = 1 - toverall / overall
        gainbucket = 1 - tbucket / bucket

    gainbucket[gainbucket<1e-6] = 0

    return gainoveral, gainbucket
   
def cal_gain_all_for_all_method(df, TYPE='C', subzero=False, N=25):

    res = {}
    for name in df.columns[N:]:
        print('='*50)
        print(name)
        if 'MV' in name or 'xgb' in name: 
            gain, gain_bucket = cal_bucket_gain_all(df, name, True, subzero=subzero, TYPE=TYPE)  
        else:
            gain, gain_bucket = cal_bucket_gain_all(df, name, False, subzero=subzero, TYPE=TYPE)

        print(gain)
        print(gain_bucket)
        gain_bucket['overall'] = gain
        res[name] = gain_bucket
    return pd.concat(res, axis=1).T


def cal_bucket_gain(df, name='delta', residual=False, TYPE='C'):
    k = df
    k = k[k['month']>36].copy()
    if TYPE == 'C':
        k['delta_bucket'] = k['delta'].apply(round_tenth)
    elif TYPE == 'P':
        k['delta_bucket'] = k['delta'].apply(round_tenth_put)

    if not residual:
        k['hedge_error'] = k['dV'] - k[name]*k['dS']
        k['hedge_error_sq'] = k['hedge_error']**2 
    else: 
        k['hedge_error_sq'] = k[name]**2

    overall = {}
    bucket = {}
    for month in k['month'].unique():
        kk = k[k['month']==month]

        overall[month] = kk['hedge_error_sq'].mean()
        bucket[month] = kk.groupby('delta_bucket')['hedge_error_sq'].mean()
    
    return pd.Series(overall), pd.concat(bucket, axis=1) 

def cal_average_bucket_gain(k, name, residual, overall, bucket, TYPE='C', mean=True):
    toverall, tbucket = cal_bucket_gain(k, name, residual, TYPE=TYPE)
    gainoveral = 1 - toverall / overall
    gainbucket = 1 - tbucket / bucket

    if not mean:
        return gainoveral, gainbucket
    return gainoveral.mean(), gainbucket.mean(axis=1)


def cal_gain_for_all_method(df, TYPE='C', N=25):
    overall, bucket = cal_bucket_gain(df, name='delta', residual=False, TYPE=TYPE)  

    for name in df.columns[N:]:
        print('='*50)
        print(name)
        
        if 'MV' in name : 
            gain, gain_bucket = cal_average_bucket_gain(df, name, True, overall, bucket, TYPE=TYPE)  
        if ('DNN' in name or 'RNN' in name) and 'MV' not in name:
            gain, gain_bucket = cal_average_bucket_gain(df, name, False, overall, bucket, TYPE=TYPE)

        print(gain)
        print(gain_bucket)
       


def generate_OTM(ticker='AAPL', TYPE='C'):
    folder = './data_'+ticker+'/'
    filename = ticker + '_Option_'+TYPE+'_A'
    df = pd.read_hdf(folder+filename+'.h5')

    if TYPE == 'C':
        suffix = '_OTM'
        df[df['delta']<=0.5] .to_hdf('./data_'+ticker+'/'+filename+suffix+'.h5', key=filename+suffix)
        suffix = '_OTM2'
        df[df['delta']<0.55] .to_hdf('./data_'+ticker+'/'+filename+suffix+'.h5', key=filename+suffix)
    elif TYPE == 'P':
        suffix = '_OTM'
        df[df['delta']>=-0.5] .to_hdf('./data_'+ticker+'/'+filename+suffix+'.h5', key=filename+suffix)
        suffix = '_OTM2'
        df[df['delta']>-0.55] .to_hdf('./data_'+ticker+'/'+filename+suffix+'.h5', key=filename+suffix)

