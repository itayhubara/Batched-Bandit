
import numpy as np
import math

def round2(x,T):
    return min(math.floor(x/2)*2,T)

def Sm(M):
    return (2-2**(-M))

def arithmetic_grid(T,M):
    return np.floor((np.linspace(0,T,M+1)[1:]/2))*2

def geometric_grid(T,M):
    results=[]
    a=2*(T/math.log(T))**(1/M)    
    for m in range(1,M+1):
        val = round2(a**m,T)
        results.append(val)
        if val==T: break
    return np.array(results)
    
def minimax_grid(T,M):
    results=[]
    a=(2*T)**(1/(2-2**(1-M)))*math.log((2*T)**(15/(2**M-1)))**(0.25-0.75/(2**M-1))
    u0=T**(1/(2-2**(1-M)))*math.log(T**(1/(2**M-1)))**(0.25-0.75/(2**M-1))
    results=[round2(u0,T)]
    uk=u0
    for i in range(1,M):
        uk_1=a*math.sqrt(uk/(math.log(2*T/uk)))
        val=round2(uk_1,T)
        results.append(val)
        uk=uk_1
        if val==T: break
    return np.array(results)

#def minimax_grid(T,M):
#    results=[]
#    aSm=(2*T)*math.log((2*T)**(15/(2**M-1)))**(Sm(M-3)/4)
#    a= aSm**(1/Sm(M-1))
#    u0=T**(1/(2-2**(1-M)))*math.log(T**(1/(2**M-1)))**(0.25-0.75/(2**M-1))
#    results=[round2(u0,T)]
#    uk=u0
#    for i in range(1,M):
#        uk_1=a*math.sqrt(uk/(math.log(2*T/uk)))
#        val=round2(uk_1,T)
#        results.append(val)
#        uk=uk_1
#        if val==T: break
#    return np.array(results)

def minmax_grid_paper(T,M):
    lt=math.log(T)
    if M==5:
        grid = [T**(16/31)*lt**(7/31),T**(24/31)*lt**(-5/31),T**(28/31)*lt**(-11/31),T**(30/31)*lt**(-14/31),T]
        return [round2(g,T) for g in grid]
        
def optimal_grid(T,delta):
    b= 256/delta**2*math.log(T*delta**2/128)
    if b<0: b=T
    for i in range(1,int(b)):
        if delta>math.sqrt(math.log(2*T/i)/i):
            break
    return [i,T]    

##Idea Itay
# take first point based on minimax and the try to estimate delta
# based on that estimation estimate tau and M define what would be the next point
#