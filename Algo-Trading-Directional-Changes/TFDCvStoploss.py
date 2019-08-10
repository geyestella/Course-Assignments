import numpy as np
import pandas as pd 

SH_index = np.genfromtxt('SH_index.csv', dtype=float,delimiter=',')
SH_index_date = np.genfromtxt('SH_index_date.csv', dtype=str,delimiter=',')
'''
for i in range(0,len(SH_index)):
    print(SH_index_date[i],"today's index is", SH_index[i])
'''

def generate(data, d,d1):
    
    p = pd.DataFrame({
    "Price": data
    })
    p["Event"] = ''
    event = 'upturn'
    ph = p['Price'][0] # highest price
    pl = ph # lowest price
    Long = np.zeros(3426,int)
    Short = np.zeros(3426,int)
    TransactIndex=np.zeros(5000)
    time = 0
    
    TotalReturn = 0
    Position = 0
    wintime = 0
    maxdrawdown = 0
    maxprofit = 0
    Profit = 0
    Loss = 0
    for i in range(0, len(p)):
        if event is 'upturn':
            if p['Price'][i] <= (ph * (1 - d1)):
                if Position == 1:
                    Long[i] = -1
                    Position = 0
                    TransactIndex[time]=p['Price'][i]
                    time=time+1
            if p['Price'][i] <= (ph * (1 - d)):
                event = 'downturn'
                pl = p['Price'][i]
                Short[i] = 1
                Position = -1
                TransactIndex[time]=p['Price'][i]
                time=time+1
            else:
                if ph < p['Price'][i]:
                    ph = p['Price'][i]
        else:
            if p['Price'][i] >= (pl * (1 + d1)):
                if Position==-1:
                    Short[i] = -1
                    Position=0
                    TransactIndex[time]=-p['Price'][i]
                    time=time+1
            if p['Price'][i] >= (pl * (1 + d)):
                event = 'upturn'
                ph = p['Price'][i]
                Long[i] = 1
                Position = 1
                TransactIndex[time]=-p['Price'][i]
                time=time+1
            else:
                if pl > p['Price'][i]:
                    pl = p['Price'][i]
    np.savetxt('TransactIndex25', TransactIndex, fmt='%f', delimiter=',')
    Return = np.zeros(time)
    j=0
    for i in range(0, time-1,2):
        Return[j]=TransactIndex[i]+TransactIndex[i+1]
        if (Return[j]>0):
            wintime=wintime+1
            Profit = Profit + Return[j]
            maxprofit=max(maxprofit,Return[i])
        else:
            maxdrawdown=min(maxdrawdown,Return[j])
            Loss = Loss + Return[j]
        TotalReturn=TotalReturn+Return[j]
        j=j+1
    '''
    print("the Return is ",Return)
    print("the transactindex is ",TransactIndex[0:time])
    np.savetxt('index525.csv', TransactIndex, fmt='%f', delimiter=',')
    '''
    print("the Transaction time and wintime is",time,wintime)
    print("At the end, the total return is",TotalReturn)
    print("the maximum drawdown is",-(maxdrawdown-maxprofit)/maxprofit)
    print("the winning rate is",wintime/time)  
    print("the profit factor is",-Profit/Loss)  
    return p['Event']
   
generate(SH_index, 0.05,0.025)





