import pandas as pd 
import numpy as np 
import argparse
import os
import time
import yfinance as yf 
import dask.dataframe as dd

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
    # liquidity indicator
    # reference : https://www.investopedia.com/ask/answers/050615/what-difference-between-open-interest-and-volume.asp#:~:text=Volume%20and%20open%20interest%20are,are%20active%2C%20or%20not%20settled.
    option = option[option['volume'] >= para['volume_threshold']]
    option = option[option['open_interest'] >= para['open_interest_threshold']]
    print('Filtering liquidity indicators, there exists {} samples'.format(option.shape[0]))
    option = option[option['cp_flag']==para['cp_flag']]

    return option

def option_feature(option, ticker, TYPE='C'):
    
    if ticker == 'SPX':
        price = pd.read_hdf(args.data_path+'SPX_Price.h5')
        price = price.set_index('date')
    else:
        price = pd.read_hdf(args.data_path+args.stock_price_file_path[:-4]+'.h5')
        price = price[price['TICKER'] == ticker].set_index('date')

    # compute moneyness
    option = option.dropna(subset=['date', 'exdate', 'impl_volatility', 'delta', 'vega', \
         'strike_price', 'best_bid', 'best_offer']).copy()
    # for robustness 
    print('After drop missing data, there exists {} samples'.format(option.shape[0]))
    option = option[option['date'].apply(lambda x: True if x in price.index else False)]
    print('After filter unseen date, there exists {} samples'.format(option.shape[0]))

    option['stock_price'] = option['date'].apply(lambda x: price.loc[x,'Price'])
    option['moneyness'] = option['stock_price'] / (option['strike_price'] /1000)

    option['price'] = (option['best_bid'] + option['best_offer'])/2 

    option['date'] = pd.to_datetime(option['date'],format = '%Y%m%d')
    option['exdate'] = pd.to_datetime(option['exdate'],format = '%Y%m%d')
    #option['last_date'] = option['last_date'].apply(int)
    #option['last_date'] = pd.to_datetime(option['last_date'],format = '%Y%m%d') 

    option['maturity'] = option['exdate'] - option['date']
    option['maturity'] = option['maturity'].apply(lambda x:x.days)
    option['maturity'] = option.apply(lambda x: x['maturity'] if x['am_settlement']==0 else x['maturity']-1, axis=1)
    option['normalized_T'] = option['maturity'] / 365

    # trade_diff, volume>0 : same things
    #option['trade_diff'] = option['date'] - option['last_date']
    #option['trade_diff'] = option['trade_diff'].apply(lambda x:x.days)
    
    option = option.drop(['best_bid', 'best_offer','exdate', 'last_date'], axis=1)  
    
    # options with less than 14 days are removed
    #option = option[option['maturity']<=365]
    option = option[option['maturity']>=14]
    print('After droping less than 14 days options, there exists {} samples'.format(option.shape[0]))

    # options whose delta less than 0.05 or greater than 0.95 are removed 
    if TYPE == 'C':
        option = option[option['delta']>=0.05 ] 
        option = option[option['delta']<=0.95 ] 
    elif TYPE == 'P':
        option = option[option['delta']<=-0.05 ] 
        option = option[option['delta']>=-0.95 ] 
    else:
        print('Wrong type')

    print('After droping options that are not in the given delta range, there exists {} samples'.format(option.shape[0]))
    # sorted to produce observations for the same option on two successive trading days
    option = option.sort_values(by=['optionid','date'])
    return option 

def drop_stock_split(df, args):
    splits = yf.Ticker(args.ticker)
    t = splits.actions['Stock Splits']
    splits = t[t>0]  
    # index is the date of stock plits
    # format : Timestamp
    df = df[df['date'].apply(lambda x: x not in list(splits.index))]
    return df 


def get_stock_price(args):
    if args.ticker == 'SPX':
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

# code for generating SPX_Price.h5
'''
p = get_stock_price('^GSPC')
p = p[p.index >= pd.Timestamp('20000101')]
p['date'] = p['date'].apply(lambda x: x.strftime('%Y%m%d'))
p['date'] = p['date'].apply(int)
p['Price'] = p['Close']
p.to_hdf('SPX_Price.h5', key='SPX_Price')
'''
def get_stock_price_yf(ticker):
    # yahoo finance symbol for SPX is '^GSPC'
    stockprice = yf.Ticker(ticker).history(period='max')
    stockprice['dS'] = stockprice['Close'].shift(-1) - stockprice['Close']
    stockprice['date'] = stockprice.index 
    stockprice['stock_day_diff'] = stockprice['date'].shift(-1) - stockprice['date'] 
    stockprice['stock_day_diff'] = stockprice['stock_day_diff'].apply(lambda x: x.days)
    return stockprice.dropna()


def cal_dVdS(df, args):
    ticker = args.ticker
    start_year = args.start_year

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

    # dV, dS is normalized so that the underlying price is one
    df['dV'] = df['dV'] / df['stock_price'] * 1000
    df['dS'] = df['dS'] / df['stock_price'] * 1000

    return df 


def cal_ret_vix(option, args):
    stockprice = get_stock_price(args)  
    stockprice['log_return'] = stockprice['Price'] / stockprice['Price'].shift(1) 
    stockprice['log_return'] = stockprice['log_return'].apply(np.log)

    stockprice['log_return_week'] = stockprice['log_return'].rolling(5).mean()
    stockprice['log_return_month'] = stockprice['log_return'].rolling(22).mean()
    # stockprice, date : Timestamp
    for name in ['log_return', 'log_return_week', 'log_return_month']: 
        option[name] = option['date'].apply(lambda x: stockprice.loc[x, name] ) 

    print('Computing vix')
    option['date_int'] = option['date'].apply(lambda x: int(x.strftime('%Y%m%d')) )
    vix = pd.read_csv(args.data_path+'VIX.csv') 
    vix = vix.set_index('Date')
    vix['vix_week'] = vix['vix'].rolling(5).mean()
    vix['vix_month'] = vix['vix'].rolling(22).mean()
    for name in ['vix', 'vix_week', 'vix_month']:
        option[name] = option['date_int'].apply(lambda x: vix.loc[x, name])
    option = option.drop(['date_int'], axis=1)

    return option

def preprocess_option(args):
    TYPE = args.cp_flag
    ticker = args.ticker + ('' if args.ticker == '' else '_')
    suffix = ('_' if args.suffix != '' else '') + args.suffix
    filename = ticker+'Option_'+ TYPE +'_' +args.exercise_style + suffix
    if not args.refresh and os.path.exists(args.data_path+filename+'.h5'):
        print(args.data_path+filename+'.h5')
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
    print('After selecting right features, there remains {} samples'.format(option.shape[0]))
    print('Do feature Engineering on Option Data')
    option = option_feature(option, args.ticker, TYPE)
    #print('Drop data when stock split happens')
    #option = drop_stock_split(option, args)
    print('Calculating dV, dS')
    option = cal_dVdS(option, args)

    print('Adding log_return, VIX')
    option = cal_ret_vix(option, args)

    print('Save processed Option Data')
    option.to_hdf(args.data_path+filename+'.h5', key=filename) 
    print("Option selection and feature engineering--- %s seconds ---" % (time.time() - start_time))

def preprocess(args):
    #preprocess_all_stock_price(args)
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
    use_cols = ['optionid', 'ticker','am_settlement', 'date', 'exdate', 'last_date', 'exercise_style', \
        'cp_flag', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest','forward_price',\
        'impl_volatility', 'delta', 'gamma', 'vega', 'theta', 'ss_flag']

    option = pd.read_csv(args.data_path+args.option_file_path, low_memory=False,\
        usecols=use_cols) 
    print('Total number of sample : {}'.format(option.shape[0]))
    print("Reading CSV files --- %s seconds ---" % (time.time() - start_time))

    print('Select option between 20100101 and 20191231')
    # after option_feature, date changes from int to Timestamp
    option = option[option['date']>=20100101]
    option = option[option['date']<20200101]
    print('After selecting date, there remains {} samples'.format(option.shape[0]))

    #option = option[option['am_settlement']==0]  # use close price for settlement
    #print('After selecting close price for settlement  {}'.format(option.shape[0])) 
    option = option[option['ss_flag']==0]      # Special Settlement, use standard settlement 
    print('After selecting standard settlement {}'.format(option.shape[0]))
    option = option.drop(['ss_flag'], axis=1)
    # filter by maturity
    '''
    if args.expiry_indicator == 'regular':
        option = option[option['expiry_indicator'].isna()] 
    else:
        option = option[option['expiry_indicator'] == 'w']
    '''
    # columns remained
    # 'optionid', 'ticker','am_settlement',
    # 'date', 'exdate', 'last_date', 'exercise_style',
    # 'cp_flag', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest',
    # 'impl_volatility', 'delta', 'gamma', 'vega', 'theta', 'forward_price',

    # forward_price may be missing
    #column_to_drop = [
    #                'secid', 'symbol', 'symbol_flag', 'cfadj', 'contract_size', 'ss_flag',
    #                'expiry_indicator', 'root', 'suffix', 'cusip',
    #                'sic', 'index_flag', 'exchange_d', 'class', 'issue_type', 
    #                'industry_group', 'issuer', 'div_convention', 'am_set_flag']
    #option = option.drop(column_to_drop, axis=1)   

    if saving:
        print('Saving Option Data into H5 file')
        option.to_hdf(args.data_path+filename+'.h5', key=filename) 
        
    return option 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='SPX')

    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--option_file_path', type=str, default='Option.csv') 
    parser.add_argument('--stock_price_file_path', type=str, default='StockPriceAll.csv') 

    parser.add_argument('--volume_threshold', type=int, default=0)
    parser.add_argument('--open_interest_threshold', type=int, default=0)
    parser.add_argument('--cp_flag', type=str, default='C') # call : 'C' or put : 'P'
    parser.add_argument('--exercise_style', type=str, default='E') # A for American, E for European
    #parser.add_argument('--expiry_indicator', type=str, default='regular') # or 'w' weekly option

    parser.add_argument('--start_year', type=int, default=2010)

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
        preprocess(args)
    elif args.mode == 'read':
        read_option_csv(args, saving=True)
    else:
        print('No such mode')

    print('='*10 + 'END' + '='*10)
    
        



   