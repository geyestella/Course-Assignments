import numpy as np
import pandas as pd 

SH_index = np.genfromtxt('SH_index.csv', dtype=float,delimiter=',')
SH_index_date = np.genfromtxt('SH_index_date.csv', dtype=str,delimiter=',')
'''
for i in range(0,len(SH_index)):
    print(SH_index_date[i],"today's index is", SH_index[i])
'''

def generate(data, d):
    
    p = pd.DataFrame({
    "Price": data
    })
    p["Event"] = ''
    event = 'upturn'
    ph = p['Price'][0] # highest price
    pl = ph # lowest price
    Buy = np.zeros(3426,int)
    Sell = np.zeros(3426,int)
    Transactiontime = 0
    Profit = 0
    Loss = 0
    maxdrawdown = 0
    maxprofit = 0
    for i in range(0, len(p)):
        if event is 'upturn':
            if p['Price'][i] <= (ph * (1 - d)):
                event = 'downturn'
                pl = p['Price'][i]
                Sell[i] = 1
                Transactiontime = Transactiontime + 1
            else:
                if ph < p['Price'][i]:
                    ph = p['Price'][i]
        else:
            if p['Price'][i] >= (pl * (1 + d)):
                event = 'upturn'
                ph = p['Price'][i]
                Buy[i] = 1
                Transactiontime = Transactiontime + 1
            else:
                if pl > p['Price'][i]:
                    pl = p['Price'][i]
        '''print('Asset is',Asset,'Total Asset is',TotalAsset)'''
    winorlose = 0
    bet = 0
    wintime = 0
    time= 0
    Return = np.zeros(Transactiontime)
    for i in range(0, len(p)-1):
        if (Buy[i]==1 or Sell[i]==1) :
            winorlose = winorlose - Buy[i] * p['Price'][i] + Sell[i] * p['Price'][i]
            bet = bet + 1
        if(bet==2):
            Return[time] = winorlose
            time= time + 1
            if (winorlose>=0):
                wintime=wintime+1
                Profit = Profit + winorlose
                maxprofit=max(maxprofit,winorlose)
            else:
                maxdrawdown=min(maxdrawdown,winorlose)
                Loss = Loss + winorlose
            bet=0
            winorlose=0
    print("the Transaction time and wintime is",Transactiontime,wintime)
    print("At the end, the total return is",Profit+Loss)
    print("the maximum drawdown is",-(maxdrawdown-maxprofit)/maxprofit)
    print("the winning rate is",wintime*2/Transactiontime)
    print("the profit factor is",-Profit/Loss)
    '''
    np.savetxt('return.csv', Return, fmt='%f', delimiter=',')
    '''
    return p['Event']
   
generate(SH_index, 0.2)


