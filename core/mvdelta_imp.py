import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class MVdelta():
    def __init__(self):
        return 
    def fit(self, df, dV=None, dS=None, progressive_bar=False):
        df = df.copy()
        df['diff'] = df['dV'] - df['delta']*df['dS']
        df['coef'] = df['vega'] * df['dS'] / (1000 * np.sqrt(df['maturity']) )
        # df['stock_price'] is replaced with 1000
        # since dS is normalized 
    
        df['scaleddelta**2'] = df['delta']**2 * df['coef']
        df['scaleddelta'] = df['delta'] * df['coef']

        X_train = df.loc[:,['coef', 'scaleddelta','scaleddelta**2']]
        y_train = df.loc[:,'diff']
        self.reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
        return 

    def predict(self, df, dV=None, dS=None):
        df = df.copy()
        df['const'] = 1
        df['delta**2'] = df['delta']**2 
        X = df.loc[:,['const', 'delta', 'delta**2']]
        dimp = self.reg.predict(X)
        dimp *= df['dS'] / 1000 / np.sqrt(df['maturity']) 
        return dimp


    def score(self, df, dV=None, dS=None):
        dimp = self.predict(df)
        loss = np.mean((dimp-df['dimp'])**2)
        return loss 