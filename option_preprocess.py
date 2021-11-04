import pandas as pd 
import numpy as np
import argparse
import os
import time
import yfinance as yf 


'''
This file deal with the preprocess of option data

1. read dataset from csv files
2. select option with suitable features 
3. generating features for future usage
    moneyness S/K, maturity T
'''


def preprocess_all_stock_price(args):
    # stock price data
    if not args.refresh and os.path.exists(args.data_path+args.stock_price_file_path[:-4]+'.h5'):
        print(args.stock_price_file_path[:-4]+'.h5' + ' already exists')
        return 
    print('Reading Stock Price Data')
    start_time = time.time()
    df = pd.read_csv(args.data_path+args.stock_price_file_path, low_memory=False)
    df['Price'] = (df['BID'] + df['ASK'])/2
    df['Spread'] = df['ASK'] - df['BID']
    df = df[['date', 'TICKER', 'Price', 'Spread', 'ASK', 'BID']] 
    #df['date'] = pd.to_datetime(df['date'])
    print('Save Stock Price Data into H5 file')
    df.to_hdf(args.data_path+args.stock_price_file_path[:-4]+'.h5', key=args.stock_price_file_path[:-4])
    print("--- %s seconds ---" % (time.time() - start_time))


def option_selection2(option, para):
    #option = option[option['open_interest']>para[volume_threshold] ] # consider option with some traded volume
    option = option[option['cp_flag']==para['cp_flag']]
    
    # liquidity indicator
    # reference : https://www.investopedia.com/ask/answers/050615/what-difference-between-open-interest-and-volume.asp#:~:text=Volume%20and%20open%20interest%20are,are%20active%2C%20or%20not%20settled.
    option = option[option['volume'] >= para['volume_threshold']]
    option = option[option['open_interest'] >= para['open_interest_threshold']]
    # Open interest is lagged by one-day after November 28th, 2000

    return option


def option_feature(option, ticker, TYPE='C'):
    
    price = pd.read_hdf(args.data_path+args.stock_price_file_path[:-4]+'.h5')
    price = price[price['TICKER'] == ticker].set_index('date')

    # compute moneyness
    # forward_price may be missing
    option = option.dropna(subset=['date', 'exdate', 'delta', 'vega', \
         'strike_price', 'best_bid', 'best_offer']).copy()
    # for robustness
    option = option[option['date'].apply(lambda x: True if x in price.index else False)]

    option['stock_price'] = option['date'].apply(lambda x: price.loc[x,'Price'])
    option['moneyness'] = option['stock_price'] / (option['strike_price'] /1000)
    option['log_forward_moneyness'] = option['forward_price'] / (option['strike_price'] /1000)
    option['log_forward_moneyness'] = option['log_forward_moneyness'].apply(np.log)

    option['price'] = (option['best_bid'] + option['best_offer'])/2 

    option['date'] = pd.to_datetime(option['date'],format = '%Y%m%d')
    option['exdate'] = pd.to_datetime(option['exdate'],format = '%Y%m%d')

    option['maturity'] = option['exdate'] - option['date']
    option['maturity'] = option['maturity'].apply(lambda x:x.days)
    option['maturity'] = option.apply(lambda x: x['maturity'] if x['am_settlement']==0 else x['maturity']-1, axis=1)
    option['normalized_T'] = option['maturity'] / 365

    option = option.drop(['best_bid', 'best_offer','exdate', 'last_date'], axis=1)  #'forward_price'
    
    # options with less than 14 days are removed
    #option = option[option['maturity']<=365]
    option = option[option['maturity']>=14]

    # options whose delta less than 0.05 or greater than 0.95 are removed 
    if TYPE == 'C':
        option = option[option['delta']>0.05 ] 
        option = option[option['delta']<=0.95 ]  
    elif TYPE == 'P':
        option = option[option['delta']<-0.05 ] 
        option = option[option['delta']>-0.95 ] 
    else:
        print('Wrong type')

    # sorted to produce observations for the same option on two successive trading days
    option = option.sort_values(by=['optionid','date'])
    return option 


def drop_stock_split(df, args):
    t = yf.Ticker(args.ticker)
    t = t.splits 
    splits = t[t>0]  
    # index is the date of stock plits
    # format : Timestamp
    if len(splits)>0:
        df = df[df['date'].apply(lambda x: x not in list(splits.index))]
    return df 

'''
def get_stock_price(args): 
    df = pd.read_hdf(args.data_path+args.stock_price_file_path[:-4]+'.h5')
    df = df[df['TICKER']==args.ticker]

    df['Close'] = df['Price']
    df['dS'] = df['Price'].shift(-1) - df['Price'] 
    df['date'] = pd.to_datetime(df['date'].apply(str))
    df['stock_day_diff'] = df['date'].shift(-1) - df['date'] 
    df['stock_day_diff'] = df['stock_day_diff'].apply(lambda x: x.days)
    return df.set_index('date').dropna()
'''

def get_stock_price(args, SPX=False):
    if SPX:
        df = pd.read_hdf(args.data_path+'SPX_Price.h5')
        df['date'] = pd.to_datetime(df['date'].apply(str))
        return df.set_index('date').dropna()
    
    df = pd.read_hdf(args.data_path+args.stock_price_file_path[:-4]+'.h5')
    df = df[df['TICKER']==args.ticker]

    df['Close'] = df['Price']
    df['dS'] = df['Price'].shift(-1) - df['Price'] 
    df['date'] = pd.to_datetime(df['date'].apply(str))
    df['stock_day_diff'] = df['date'].shift(-1) - df['date'] 
    df['stock_day_diff'] = df['stock_day_diff'].apply(lambda x: x.days)
    return df.set_index('date').dropna()


def cal_dVdS(df, args, start_year=2010):
    ticker = args.ticker

    df['month'] = df['date'].apply(lambda x: (x.year-start_year)*12+x.month)
    stockprice = get_stock_price(args)  
    df['stock_price'] = df['date'].apply(lambda x: stockprice.loc[x,'Close'])

    df['dS'] = df['date'].apply(lambda x: stockprice.loc[x,'dS'])
    df['stock_day_diff'] = df['date'].apply(lambda x: stockprice.loc[x,'stock_day_diff'])

    df = df.sort_values(['optionid', 'date'])
    
    df['dV'] = df.groupby('optionid').apply(lambda x:x['price'].shift(-1) - x['price'])\
        .reset_index().set_index('level_1')['price']
    df['dimp'] = df.groupby('optionid').apply(lambda x:x['impl_volatility'].shift(-1) - x['impl_volatility'])\
        .reset_index().set_index('level_1')['impl_volatility']
    df['option_day_diff'] = df.groupby('optionid').apply(lambda x:x['date'].shift(-1) - x['date'])\
        .reset_index().set_index('level_1')['date']
    df['option_day_diff'] = df['option_day_diff'].apply(lambda x: x.days)
    df = df[df['option_day_diff'] == df['stock_day_diff'] ]
    df = df.drop(['option_day_diff', 'stock_day_diff'], axis=1)

    # dV, dS is normalized so that the underlying price is 1000
    df['dV'] = df['dV'] / df['stock_price'] * 1000
    df['dS'] = df['dS'] / df['stock_price'] * 1000
    return df 


def preprocess_option(args):
    TYPE = args.cp_flag
    ticker = args.ticker + ('' if args.ticker == '' else '_')
    suffix = ('_' if args.suffix != '' else '') + args.suffix
    filename = ticker+'Option_'+ TYPE +'_'+args.exercise_style + suffix

    if not args.refresh and os.path.exists(args.data_path+filename+'.h5'):
        print(args.data_path+ticker+filename+'.h5')
        return 

    start_time = time.time()
    option = read_option_csv(args, saving=False)

    print('Selecting Option Data')
    para_dict = {
        'volume_threshold':args.volume_threshold,  
        'open_interest_threshold': args.open_interest_threshold,          
        'cp_flag':args.cp_flag.upper()
    }
    option = option_selection2(option, para=para_dict)

    print('Do feature Engineering on Option Data')
    option = option_feature(option, args.ticker, TYPE)

    print('Drop data when stock split happens')
    option = drop_stock_split(option, args)

    print('Calculating dV, dS')
    option = cal_dVdS(option, args)

    print('Save processed Option Data')
    option.to_hdf(args.data_path+filename+'.h5', key=filename) 
    print("Option selection and feature engineering--- %s seconds ---" % (time.time() - start_time))


def preprocess(args):
    preprocess_all_stock_price(args)
    preprocess_option(args)

    return 


def read_option_csv(args, saving=False):
    ticker = args.ticker + ('' if args.ticker == '' else '_')
    suffix = ('_' if args.suffix != '' else '') + args.suffix
    filename = ticker + 'Option'+ suffix

    if not args.refresh and os.path.exists(args.data_path+filename+'.h5'):
        print(args.data_path+filename+'.h5' +' already exists')
        return pd.read_hdf(args.data_path+filename+'.h5')

    print('Reading Option Data (CSV)')
    start_time = time.time()
    use_cols = ['optionid', 'ticker','am_settlement', 'date', 'exdate', 'last_date', \
        'exercise_style', 'expiry_indicator','ss_flag', 'forward_price',\
        'cp_flag', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest',\
        'impl_volatility', 'delta', 'gamma', 'vega', 'theta']
    option = pd.read_csv(args.data_path+args.option_file_path, low_memory=False,\
        usecols=use_cols) 
    print('Total number of sample : {}'.format(option.shape[0]))
    print("Reading CSV files --- %s seconds ---" % (time.time() - start_time))

    option = option[option['date']>=20100101]
    option = option[option['date']<=20191231]

    #option = option[option['am_settlement']==0]  # use close price for settlement 
    option = option[option['ss_flag']==0]      # use standard settlement 
    #option = option[option['symbol_flag']==1]  # use new OSI data
    option = option[option['exercise_style'] == args.exercise_style]  
    
    if args.expiry_indicator == 'regular':
        option = option[option['expiry_indicator'].isna()] 
    else:
        option = option[option['expiry_indicator'] == 'w']
   
    option = option.drop(['expiry_indicator','ss_flag'], axis=1)

    #'cp_flag', 'symbol_flag','exercise_style', 'forward_price', 'am_settlement'
    #column_to_drop = ['am_set_flag','ss_flag',
    #                  'secid','symbol', 'root','suffix','issuer', 'expiry_indicator',
    #                  'cusip', 'sic', 'index_flag', 'exchange_d',
    #                  'class', 'issue_type', 'industry_group', 'div_convention',
    #                  'cfadj', 'contract_size',	]
    #option = option.drop(column_to_drop, axis=1)   

    if saving:
        print('Saving Option Data into H5 file')
        option.to_hdf(args.data_path+filename+'.h5', key=filename) 
        
    return option 


def combine_sentiment(args):
    TYPE = args.cp_flag
    ticker = args.ticker + ('' if args.ticker == '' else '_')
    filename = ticker+'Option_'+ TYPE +'_'+args.exercise_style

    if not os.path.exists(args.data_path+filename+'.h5'):
        print(args.data_path+ticker+filename+'.h5 doesn\'t exist')
        return 
    
    start_time = time.time()
    print('Starting reading option and news information')
    print('Option file : '+args.data_path+filename+'.h5')
    option = pd.read_hdf(args.data_path+filename+'.h5')

    news_filename = ticker + 'News'
    print('News file : '+args.data_path+news_filename+'.h5')
    news = pd.read_hdf(args.data_path+news_filename+'.h5')
    
    # trade date 
    print('Compute the date')
    k = pd.read_hdf(args.data_path+'StockPriceAll.h5')
    first_date = news['RPNA_DATE_UTC'].min()
    last_date = news['RPNA_DATE_UTC'].max()
    k = k[(k['TICKER']==args.ticker) & (k['date'] >= first_date) & (k['date'] <= last_date)]
    print('Number of trading dates : {}'.format(k.shape[0]))
    idx = k['date']
    # to speed up, each date is computed only once and saved in dictionary
    date_map = {}
    for dt in news['RPNA_DATE_UTC'].unique():
        date_map[dt] = idx[idx>=dt].iloc[0] 
    print('Number of dates : {}'.format(len(date_map)))
    news['date'] = news['RPNA_DATE_UTC'].map(date_map)
    '''
    r = 0
    for key in date_map:
        if r >= 10:
            break
        print(key, date_map[key])
        r += 1
    '''
    print('Incorporating information of news item into day sentiment')
    sentiment = news.groupby('date')['CSS'].apply(lambda x: (x[x>50].count(), x[x<50].count(), x[x==50].count(), x.count()))

    sentiment = pd.DataFrame(sentiment.to_list(), index=sentiment.index, \
        columns=['pos_count', 'neg_count', 'neu_count', 'count'])

    sentiment['max_CSS'] = news.groupby('date')['CSS'].apply(lambda x: (x.max()-50)/50)
    sentiment['min_CSS'] = news.groupby('date')['CSS'].apply(lambda x: (x.max()-50)/50)
    
    sentiment['high_CSS'] = news.groupby('date')['CSS'].apply(lambda x: (x[x>70].mean()-50)/50 if x[x>70].shape[0]>0 else 0)
    sentiment['low_CSS'] = news.groupby('date')['CSS'].apply(lambda x: (x[x<30].mean()-50)/50 if x[x<30].shape[0]>0 else 0)
    
    #sentiment.index = pd.to_datetime(sentiment.index, format='%Y%m%d')
    sentiment['day_CSS'] = (sentiment['pos_count'] - sentiment['neg_count'])/(sentiment['count']-sentiment['neu_count'])

    # in some case, there may be no news in a day
    # we fill it with sentiment of previous day
    sentiment = sentiment.reindex(k.set_index('date').index)
    sentiment = sentiment.fillna(method='ffill')
    #print(sentiment.head())
    print('Combine sentiment into option data')
    for name in ['max', 'min', 'day', 'high', 'low']:
        print(name+'_CSS')
        option[name+'_CSS'] = option['date'].apply(lambda x: sentiment.loc[int(x.strftime('%Y%m%d')), name+'_CSS'] ) 

    print('Save Option Data that combines sentiment')
    option.to_hdf(args.data_path+filename+'.h5', key=filename) 
    print("Combining sentiment --- %s seconds ---" % (time.time() - start_time))
    return 


def combine_variables(args):
    TYPE = args.cp_flag
    ticker = args.ticker + ('' if args.ticker == '' else '_')
    filename = ticker+'Option_'+ TYPE +'_'+args.exercise_style

    if not os.path.exists(args.data_path+filename+'.h5'):
        print(args.data_path+ticker+filename+'.h5 doesn\'t exist')
        return 
    
    start_time = time.time()
    print('Starting reading option information')
    print('Option file : '+args.data_path+filename+'.h5')
    option = pd.read_hdf(args.data_path+filename+'.h5')
    variables = pd.read_hdf(args.data_path+'StockPriceAll.h5')
    #elif args.ticker == 'semiconduct':
    #    variables = pd.read_hdf(args.data_path+'stockprice_semiconduct.h5')
    #else:
    #    print('Wrong ticker : {}'.format(args.ticker))
    variables = variables[variables['TICKER'] == args.ticker]

    variables['Price_adj_backward'] = variables['Price']
    hist = yf.Ticker(args.ticker).history(period="max")
    hist = hist.reset_index()
    hist['date'] = hist['Date'].apply(lambda x: int(x.strftime('%Y%m%d')))
    variables['Price_adj_backward'] = variables['date'].apply(lambda x: hist.set_index('date').loc[x, 'Close'])
    variables = variables.set_index('date')
    variables['log_return'] = variables['Price_adj_backward'] / variables['Price_adj_backward'].shift(1) 
    variables['log_return'] = variables['log_return'].apply(np.log)
    variables['log_return_week'] = variables['log_return'].rolling(5).mean()
    variables['log_return_month'] = variables['log_return'].rolling(22).mean()

    variables['vol_22'] = variables['log_return'].rolling(22).std()*np.sqrt(22)
    print('Computing log_return, vol_22')
    option['date_int'] = option['date'].apply(lambda x: int(x.strftime('%Y%m%d')) )
    for name in ['log_return', 'log_return_week', 'log_return_month', 'vol_22']: #'vix'
        option[name] = option['date_int'].apply(lambda x: variables.loc[x, name] ) 

    # get SPX log return 
    print('Computing SPX log_return')
    stockprice = get_stock_price(args, SPX=True)  
    stockprice['SPX_log_return'] = stockprice['Price'] / stockprice['Price'].shift(1) 
    stockprice['SPX_log_return'] = stockprice['SPX_log_return'].apply(np.log)
    # stockprice, date : Timestamp
    for name in ['SPX_log_return']: 
        option[name] = option['date'].apply(lambda x: stockprice.loc[x, name] ) 

    print('Computing vix')
    vix = pd.read_csv(args.data_path+'VIX.csv')    
    vix['vix_week'] = vix['vix'].rolling(5).mean()
    vix['vix_month'] = vix['vix'].rolling(22).mean()

    vix = vix.set_index('Date')
    for name in ['vix', 'vix_week', 'vix_month']:
        option[name] = option['date_int'].apply(lambda x: vix.loc[x, name])
    option = option.drop(['date_int'], axis=1)

    print('Save Option Data that combines variables')
    option.to_hdf(args.data_path+filename+'.h5', key=filename) 
    print("Combining variables --- %s seconds ---" % (time.time() - start_time))
    return 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='All')

    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--option_file_path', type=str, default='Option.csv') 
    parser.add_argument('--stock_price_file_path', type=str, default='StockPriceAll.csv') 

    parser.add_argument('--volume_threshold', type=int, default=0)
    parser.add_argument('--open_interest_threshold', type=int, default=0)
    parser.add_argument('--cp_flag', type=str, default='C') # call : 'C' or put : 'P'
    parser.add_argument('--exercise_style', type=str, default='A') # American
    parser.add_argument('--expiry_indicator', type=str, default='regular') # or 'w' weekly option

    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--refresh', action='store_true') # default false

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    if args.ticker != '':
        args.data_path = args.data_path[:-1] + '_' + args.ticker + '/'
        args.option_file_path = args.ticker + '_' + args.option_file_path

    print('='*9 + 'START' + '='*9)
    print(args)
    if args.mode == 'all':    
        if args.ticker == 'All':
            
            print('Starting preprocess option data of all stocks')
            read_option_csv(args, saving=True)

            df = pd.read_hdf('../data_All/All_Option.h5')
            print('Split All Option Data by Stock')

            for ticker in df['ticker'].unique():
                t = df[df['ticker']==ticker]
                filename = '../data_All/'+ticker+'_Option.h5'
                if args.refresh or not os.path.exists(filename):
                    t.to_hdf(filename, key=ticker)
                    print('{} Saved, number of samples : {}'.format(ticker, t.shape[0]))
                else:
                    print(filename + ' already exists')
            print('Total {} stocks'.format(len(df['ticker'].unique())))
            

            print('Preprocess All Stock Option Data')
            for f in os.listdir(args.data_path):
                idx = f.find("_Option.h5")
                if  idx != -1:
                    if 'All' in f:
                        continue
                    print(f)
                    args.ticker = f[:idx]
                    args.option_file_path = f
                    
                    preprocess(args)
                    combine_variables(args)

            print('Concat All Stock Option into One')
            All_Option_A = []
            for f in os.listdir('../data_All'):
                idx = f.find("_Option_"+args.cp_flag+"_A.h5")
                if idx != -1:
                    if 'All' in f:
                        continue
                    df = pd.read_hdf('../data_All/'+f)
                    print(f, df.shape[0])
                    All_Option_A.append(df)
            pd.concat(All_Option_A).to_hdf('../data_All/All_Option_' +args.cp_flag+'_A.h5', key='all')
            print('All Stock Option Data is Saved')
            
        else:
            preprocess(args)
            combine_variables(args)
    
    elif args.mode == 'read':
        read_option_csv(args, saving=True)
    elif args.mode == 'combine':
        combine_sentiment(args)
    elif args.mode == 'cv':
        combine_variables(args)


    else:
        print('No such mode')

    print('='*10 + 'END' + '='*10)
    
        



   