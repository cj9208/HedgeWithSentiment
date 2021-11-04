from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import pandas as pd
import scipy

from tqdm import tqdm

def loss_fn(delta_pred, dV, dS, d=2):
    # delta_pred, dV, dS should be tensor of the same size
    t = dV - delta_pred * dS 
    return np.mean(t**d)


def spline_generate(df, N, d, name, left=0, right=1):
    # df: pd.Series

    length = right - left
    dt = length / N 
    t_list = [left + i*dt for i in range(1,N+1)]

    dfret = {}

    for power in range(d+1):
        dfret[name+str(power)] = df.apply(lambda x: ((x-left)/length)**power)

    for t in t_list:
        dfret[name+str(t)] = df.apply(lambda x: (max(x-t, 0)/length)**d)

    return pd.concat(dfret, axis=1)


def spline_map(df, N=10, d=1, volatility=False, bucket=1, TYPE='C', sentiment=False):
    # df contain original feature, moneyness / maturity
    # use spline function
    # moneyness : around 1, choose [0,2], divided by 2*N
    # maturity : normalized, [0,1], divided by N
    # sentiment : normalized, normal distribution, [-2,2], divided by N
    
    dfret = {}

    dfret['T'] = spline_generate(df['normalized_T'], N, d, 'T', left=0, right=1)
    dfret['money'] = spline_generate(df['moneyness'], N, d, 'money', left=0.5, right=1.5)
    if volatility:
        if TYPE == 'C':    
            #dfret['volatility'] = spline_generate(df['delta'], N, d, 'delta', left=-0.05+bucket/10, right=0.05+bucket/10)
            # why include bucket worse the performance?
            dfret['volatility'] = spline_generate(df['delta'], N, d, 'delta', left=0, right=1)
        elif TYPE == 'P':    
            dfret['volatility'] = spline_generate(df['delta'], N, d, 'delta', left=-0.05-bucket/10, right=0.05-bucket/10)
        else:
            print('Wrong Option Type')

    if sentiment:
        dfret['sentiment'] = spline_generate(df['normalized_CSS'], N, d, 'sentiment', left=-1, right=1)
    return dfret
  

def kernel_map(X, Y=None, multiple=True):
    # X: N*F (N: number of samples, F: number of features)
    # Y: if None, kernel with itself

    # Generalization: can deal with dict
    # X, Y is dict of the same size
    if Y is None:
        Y = X

    res = None
    for name in X.keys():
        if res is None:
            res = np.dot(X[name], np.transpose(Y[name]))
        else:
            if multiple:
                res *= np.dot(X[name], np.transpose(Y[name]))
            else:
                res += np.dot(X[name], np.transpose(Y[name]))
    return res 


class NianHedge(BaseEstimator, ClassifierMixin):
    def __init__(self, lamb=1, kernel='spline', N=10, d=1, loss_d=2, volatility=False, bucket=1, TYPE='C', 
                    sentiment=False, multiple=True, debug=False):
        self.hedge = 'direct'
        self.volatility = volatility
        self.bucket = bucket
        self.TYPE = TYPE

        self.sentiment = sentiment
        self.kernel = kernel
        self.N = N
        self.d = d
        self.multiple=multiple
        self.lamb = lamb
        self.loss_d = loss_d
        self.debug = debug
    
    def get_feature(self, df_feature):
        df_feature = df_feature.copy()
        feature_list = ['moneyness','normalized_T']
        if self.volatility is True:
            feature_list.append('delta')

        if self.sentiment is True :
            feature_list.append('normalized_CSS')
            self.CSS_mean = np.mean(df_feature['normalized_CSS'])
            self.CSS_std = np.std(df_feature['normalized_CSS'])
            df_feature['normalized_CSS'] = (df_feature['normalized_CSS']-self.CSS_mean)/self.CSS_std

        df_feature = df_feature[feature_list]
        return df_feature

    def fit(self, df_feature, dV, dS):
        '''
        df_feature : N * F
        dV, dS : N 
        ''' 
        df_feature = self.get_feature(df_feature)
        if self.kernel == 'spline':
            df_feature = spline_map(df_feature, N=self.N, d=self.d, volatility=self.volatility, bucket=self.bucket,
                                                TYPE=self.TYPE, sentiment=self.sentiment)

        self.df_feature = df_feature
        
        K = kernel_map(df_feature, multiple=self.multiple)
        D = np.diag(dS)
        Ktide = np.matmul(D, K) 

        # the bottleneck lies at the computation of inverse matrix
        # I test with simple matrix and find th at with symmetrix matrix, 
        # the time complexity of both inv and svd is N^3
        # however, the constant associated with inv is smaller than that of svd
        
        # use the acceleration in the paper
        # Q: what is the quick way of getting eigen decomposition?
        #s = (s ** 2 + self.lamb)**(-1)
        #t1 = np.dot(np.transpose(vh), np.diag(s))
        #t1 = np.dot(t1, vh)
        Klast = np.matmul(np.transpose(Ktide), Ktide) + self.lamb * np.eye(Ktide.shape[0])
        t1 = scipy.linalg.inv( Klast )   
        
        t2 = np.dot(np.transpose(Ktide), dV) #N*1
        self.alpha = np.dot(t1,t2) #N*1
        
        #for debug
        if self.debug:
            print(self.alpha.shape)
            print(self.lamb) 
            print(Ktide[:3,:3])
            print(t1[:3,:3])
            
        # solving linear equation Ax = b
        # should not use x = inv(A)*b, use A \ b  
        # see https://www.mathworks.com/help/matlab/ref/mldivide.html for explanation of the operator
        # inv vs mldivide https://stackoverflow.com/questions/1419580/why-is-matlabs-inv-slow-and-inaccurate
        # inv vs pinv https://stackoverflow.com/questions/19423198/why-is-the-output-of-inv-and-pinv-not-equal-in-matlab-and-octave
        # linalg (contain comparision of scipy.linalg and np.linalg, difference of pinv and pinv2(two implementation)) 
        #       https://scipy.github.io/devdocs/tutorial/linalg.html
        # lstsq vs mldivide https://stackoverflow.com/questions/33559946/numpy-vs-mldivide-matlab-operator
        # np.linalg.lstsq https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
        #b = np.dot(Ktide, dV)
        #self.alpha = np.linalg.lstsq(Klast,b, rcond=None)[0]
     
         
    def predict(self, df_feature_test, kernel='spline'):
        df_feature_test = self.get_feature(df_feature_test)
        if self.kernel == 'spline':
            df_feature_test = spline_map(df_feature_test, N=self.N, d=self.d, volatility=self.volatility,sentiment=self.sentiment)
        K = kernel_map(self.df_feature, df_feature_test, multiple=self.multiple) # N * N_test
        return np.dot(self.alpha, K)

    def score(self, df_feature, dV, dS):
        delta = self.predict(df_feature)
        return loss_fn(delta, dV, dS, d=self.loss_d)


class NianwithGridSearch():
    def __init__(self, Nian, lamb_list=None):
        self.classifier = Nian 
        self.lamb_list = lamb_list


    def fit(self, df_feature, dV, dS, progressive_bar=True):
        if self.lamb_list is None:
            return 
        
        best_lamb = None
        best_score = None

        if progressive_bar:
            iter_progressive = tqdm(self.lamb_list,desc='Searching for best lamb : ')
        else:
            iter_progressive = self.lamb_list
        for lamb in iter_progressive:
            self.classifier.lamb = lamb
            self.classifier.fit(df_feature, dV, dS)
            if best_score is None:
                best_lamb = lamb
                best_score = self.classifier.score(df_feature, dV, dS)
            else:
                score = self.classifier.score(df_feature, dV, dS)
                if score < best_score:
                    best_lamb = lamb
                    best_score = score

        self.classifier.lamb = best_lamb
        return self.classifier

    def predict(self, df_feature):
        return self.classifier.predict(df_feature)
    

    def score(self, df_feature, dV, dS):
        delta = self.classifier.predict(df_feature)
        return loss_fn(delta, dV, dS)


def round_tenth(x):
    # 0.05 <= x <= 0.95 
    for i in range(1,10):
        if x >=-0.05 + i/10 and x < 0.05 + i/10:
            return i 
    return 9  

class NianDeltaBucket():
    def __init__(self, para):
        self.para = para

        self.classifier = {}
        base_model = {}
        for i in range(1,9+1):
            base_model[i] = NianHedge(lamb=self.para['lamb'], N=self.para['N'], 
                    d=self.para['d'], loss_d=self.para['loss_d'], volatility=self.para['volatility'], bucket=i,
                    TYPE = self.para['TYPE'], multiple=self.para['multiple'])   
            self.classifier[i] = NianwithGridSearch(base_model[i], 10.0**np.arange(-self.para['lamb_max'],self.para['lamb_max']+1) )


    def fit(self, df_feature, dV, dS, progressive_bar=False):
        df = df_feature.copy()
        df['bucket'] = df['delta'].apply(round_tenth)

        for i in range(1,9+1):
            idx = df['bucket']==i

            df_part = df[idx]
            dV_part = dV[idx]
            dS_part = dS[idx]

            #print(df_part['delta'].min(), df_part['delta'].max())
            self.classifier[i].fit(df_part, dV_part, dS_part, progressive_bar=progressive_bar)

    def predict(self, df_feature):
        df = df_feature.copy()
        df['bucket'] = df['delta'].apply(round_tenth)

        res = []
        for i in range(1,9+1):
            idx = df['bucket']==i
            df_part = df[idx]

            valid_delta = self.classifier[i].predict(df_part)
            res.append(pd.Series(valid_delta, index=df_part.index) )

        return pd.concat(res)

    def score(self, df_feature, dV, dS):
        df = df_feature.copy()
        df['bucket'] = df['delta'].apply(round_tenth)

        all_valid_loss = []
        all_valid_num = []

        for i in range(1,9+1):
            idx = df['bucket']==i

            df_part = df[idx]
            dV_part = dV[idx]
            dS_part = dS[idx]

            valid_loss = self.classifier[i].score(df_part, dV_part, dS_part)
            valid_num = df_part.shape[0]

            all_valid_loss.append(valid_loss)
            all_valid_num.append(valid_num)

        return np.average(all_valid_loss, weights=all_valid_num/np.sum(all_valid_num))


