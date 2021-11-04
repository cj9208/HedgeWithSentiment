import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

para_default = {
    'booster': 'gbtree',
    'objective':'reg:squarederror',
    'eval_metric':'rmse',
    'max_depth':6,
    'eta':0.1,
    'gamma':0,
    'min_child_weight':1,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'lambda':1,
    'alpha':0,
    'nthread':-1,
    'seed':0,
    'verbosity':0
}


class xgboostdelta():
    def __init__(self, vix=True, para=para_default):
        self.para = para 
        self.vix = vix
        self.feature_list = ['moneyness', 'normalized_T', 'delta']
        if self.vix:
            self.feature_list.extend(['impl_volatility', 'log_return', 'vol_22', 'vix'] )
        #self.classifier = xgb.XGBRegressor(**self.para)
        return 

    def fit(self, df, dV=None, dS=None, progressive_bar=False):
        df = df.copy()
        df_feature = df[self.feature_list]
        df['diff'] = df['dV'] - df['delta']*df['dS']

        dtrain = xgb.DMatrix(df_feature, label=df['diff'])
   
        # 超参调节
        #cv_res= xgb.cv(self.para, dtrain, num_boost_round=100, early_stopping_rounds=5, nfold=5, metrics='rmse')
        # cv_res.shape[0] 为最佳迭代次数
        #bst = xgb.train(self.para, dtrain,num_boost_round=cv_res.shape[0])
        #self.classifier = bst 

        
        x_train, x_valid, y_train, y_valid = train_test_split(df_feature, df['diff'], test_size=0.2)

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_valid, label=y_valid) 
        evallist  = [(dtrain, 'train'), (dvalid, 'eval')]

        cv_res= xgb.cv(self.para, dtrain, num_boost_round=100, early_stopping_rounds=5, nfold=5, metrics='rmse')

        n_estimators = [cv_res.shape[0]]
        max_depth = [2, 4, 6]
        learning_rate=[0.05, 0.1, 0.20]
        min_child_weight=[1,2,4]
        lambdas = [1,2,5]

        hyperparameter_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate':learning_rate,
            'min_child_weight':min_child_weight,
            'lambda': lambdas
        }

        random_cv = RandomizedSearchCV(estimator=xgb.XGBRegressor(**self.para),
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50, n_jobs = -1,
            verbose = 0,  # no output
            return_train_score = True,
            random_state=2)

        random_cv.fit(df_feature, df['diff'])

        self.para.update(random_cv.best_params_)
        self.para.pop('n_estimators', None)

        self.classifier = xgb.train(self.para, dtrain, num_boost_round=200, early_stopping_rounds=20,\
            evals=evallist, verbose_eval=False)
        
        return 

    def predict(self, df, dV=None, dS=None):
        df = df.copy()
        df_feature = df[self.feature_list]
        dtest = xgb.DMatrix(df_feature)

        df['diff'] = df['dV'] - df['delta']*df['dS']
        df['xgb_modification'] = self.classifier.predict(dtest)
        df['diff_modified'] = df['diff'] - df['xgb_modification'] 
        return df['diff_modified']


    def score(self, df, dV=None, dS=None):
        loss = np.mean(self.predict(df)**2)
        return loss 