""""""                                                                                                               
"""                                                                                                               
Template for implementing StrategyLearner  (c) 2016 Tucker Balch                                                                                                               
                                                                                                               
Copyright 2018, Georgia Institute of Technology (Georgia Tech)                                                                                                               
Atlanta, Georgia 30332                                                                                                               
All Rights Reserved                                                                                                               
                                                                                                               
Template code for CS 4646/7646                                                                                                               
                                                                                                               
Georgia Tech asserts copyright ownership of this template and all derivative                                                                                                               
works, including solutions to the projects assigned in this course. Students                                                                                                               
and other users of this template code are advised not to share it with others                                                                                                               
or to make it available on publicly viewable websites including repositories                                                                                                               
such as github and gitlab.  This copyright statement should not be removed                                                                                                               
or edited.                                                                                                               
                                                                                                               
We do grant permission to share solutions privately with non-students such                                                                                                               
as potential employers. However, sharing with other current or future                                                                                                               
students of CS 7646 is prohibited and subject to being investigated as a                                                                                                               
GT honor code violation.                                                                                                               
                                                                                                               
-----do not edit anything above this line---                                                                                                               
                                                                                                               
Student Name: Tucker Balch (replace with your name)                                                                                                               
GT User ID: tb34 (replace with your User ID)                                                                                                               
GT ID: 900897987 (replace with your GT ID)                                                                                                               
"""                                                                                                               
                                                                                                               
import datetime as dt                                                                                                               
import random                                                                                                                                                                                                                              
import pandas as pd                                                                                                               
import util as ut
import BagLearner as bl
import RTLearner as rt
import indicators 
from indicators import *  
import numpy as np                                                                                                             
                                                                                                               
                                                                                                               
class StrategyLearner(object):                                                                                                               
    """                                                                                                               
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.                                                                                                               
                                                                                                               
    :param verbose: If “verbose” is True, your code can print out information for debugging.                                                                                                               
        If verbose = False your code should not generate ANY output.                                                                                                               
    :type verbose: bool                                                                                                               
    :param impact: The market impact of each transflag, defaults to 0.0                                                                                                               
    :type impact: float                                                                                                               
    :param commission: The commission amount charged, defaults to 0.0                                                                                                               
    :type commission: float                                                                                                               
    """  
    def author(self):
        return 'ydong336'  
                                                                                                         
    # constructor                                                                                                               
    def __init__(self, verbose=False, impact=0.0, commission=0.0):                                                                                                               
        """                                                                                                               
        Constructor method                                                                                                               
        """                                                                                                               
        self.verbose = verbose                                                                                                               
        self.impact = impact                                                                                                               
        self.commission = commission        
        self.learner = bl.BagLearner(learner = rt.RTLearner,kwargs={"leaf_size":5}, bags = 20, boost = False, verbose = False)
        self.ndays=5                                                                                                                 
                                                                                                               
    # this method should create a QLearner, and train it for trading                                                                                                               
    def add_evidence(                                                                                                               
        self,                                                                                                               
        symbol="IBM",                                                                                                               
        sd=dt.datetime(2008, 1, 1),                                                                                                               
        ed=dt.datetime(2009, 1, 1),                                                                                                               
        sv=10000,                                                                                                               
    ):                                                                                                               
        """                                                                                                               
        Trains your strategy learner over a given time frame.                                                                                                               
                                                                                                               
        :param symbol: The stock symbol to train on                                                                                                               
        :type symbol: str                                                                                                               
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008                                                                                                               
        :type sd: datetime                                                                                                               
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009                                                                                                               
        :type ed: datetime                                                                                                               
        :param sv: The starting value of the portfolio                                                                                                               
        :type sv: int                                                                                                               
        """                                                                                                               
                                                                                                               
        # add your code to do learning here                                                                                                               
                                                                                                               
        # example usage of the old backward compatible util function                                                                                                               
        syms = [symbol]                                                                                                               
        dates = pd.date_range(sd, ed)                                                                                                               
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY                                                                                                               
        prices = prices_all[syms]  # only portfolio symbols                                                                                                               
        #prices_SPY = prices_all["SPY"]  # only SPY, for comparison later                                                                                                               
        if self.verbose:                                                                                                               
            print(prices)                                                                                                               
                                                                                                               
        # example use with new colname                                                                                                               
        # volume_all = ut.get_data(                                                                                                               
        #     syms, dates, colname="Volume"                                                                                                               
        # )  # automatically adds SPY                                                                                                               
        # volume = volume_all[syms]  # only portfolio symbols                                                                                                               
        # volume_SPY = volume_all["SPY"]  # only SPY, for comparison later                                                                                                               
        # if self.verbose:                                                                                                               
        #     print(volume)          
        window_size=20
        sma = get_price_by_SMA(prices,window_size)
        bollinger = get_bollinger_bands(prices,window_size)
        momentum=get_momentum(prices,window_size)
        
        trainX= pd.concat((sma,bollinger,momentum),axis=1)
        trainX.fillna(0,inplace=True)
        trainX= trainX[:-self.ndays]

        threshold= ((prices.values[self.ndays:]/prices.values[:-self.ndays])-1).T[0]
        sell_threshold= -0.02 - self.impact
        buy_threshold= 0.02 + self.impact
        
        sell= (threshold < sell_threshold).astype(int)
        buy= (threshold > buy_threshold).astype(int)
        
        trainY=np.array(buy-sell)
        trainX= trainX.values
       
        self.learner.add_evidence(trainX,trainY)  
        
    # this method should use the existing policy and test it against new data                                                                                                                  
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):                                                                                                                  
                                                                                                                  
        syms=[symbol]                                                                                                                  
        dates = pd.date_range(sd, ed)                                                                                                                  
        prices_all = ut.get_data(syms, dates)                                                                                                                  
        prices = prices_all[syms]                                                                                                                                  
        if self.verbose: print(prices)

        window_size=20
        sma = get_price_by_SMA(prices,window_size)
        bollinger = get_bollinger_bands(prices,window_size)
        momentum=get_momentum(prices,window_size)

        testX= pd.concat((sma,bollinger,momentum),axis=1)
        testX.fillna(0,inplace=True)
        testX= testX.values
        testY= self.learner.query(testX)
        
        trades = np.zeros(testY.shape[0])
        flag=0
        rows=testY.shape[0]
        for i in range(0,rows-1):            
            if(testY[i]<0):
                if(flag==0):
                    trades[i]= -1000
                elif(flag==1):
                    trades[i]= -2000
                flag= -1
            elif(testY[i]>0):
                if(flag==0):
                    trades[i]= 1000
                elif(flag==-1):
                    trades[i]= 2000
                flag= 1
                
        if(flag==1):
            trades[-1]=-1000
        elif(flag==-1):
            trades[-1]=1000
        

        trades=pd.DataFrame(trades, index=prices.index)
        return trades                                                                                                                  
                                                                                                                  
if __name__=="__main__":                                                                                                                  
    print("One does not simply think up a strategy")              
                
        