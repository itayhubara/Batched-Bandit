from scipy.stats import bernoulli
from scipy.stats import norm
import math
import numpy as np
from sampler import Sampler

class UCB2(Sampler):
    """
    This class implement ucb2
    """
    def __init__(self,alpha,delta,optimal_arm,dist_type):
        super(UCB2, self).__init__(alpha,delta,optimal_arm,dist_type)
        self.alpha=alpha
        #self.n=0
        self.r=[0,0]
        self.num_batches=0

    def tau(self,r):
        return math.ceil((1+self.alpha)**r)

    def anr(self,r): 
        ''' rethurn the anr as appear in the paper '''
        return math.sqrt(((1+self.alpha)*math.log(math.exp(1)*self.t/self.tau(r)))/(2*self.tau(r)))   
    
    def select_arm_update(self):
        """
        This function implement the main par tof ucb2
        """
        if self.t==0:
            self.rewards_list[self.optimal_arm].append(self.optimal_sampler.rvs(1))
            self.rewards_list[abs(self.optimal_arm-1)].append(self.dagger_sampler.rvs(1))   
            self.t = 2
            return
        else:
            x0=sum(self.rewards_list[0])/len(self.rewards_list[0])
            x1=sum(self.rewards_list[1])/len(self.rewards_list[1])
            anr0 = self.anr(self.r[0])
            anr1 = self.anr(self.r[1])
            arm = np.argmax([x0+anr0,x1+anr1])
            batch_size=self.tau(self.r[arm]+1)-self.tau(self.r[arm])
            self.r[arm]+=1
            self.num_batches+=1
            return arm,batch_size    

    def sample_reward(self,arm):
        """
        This function will return a stochastic reward
        """ 
        self.t+=1
        if arm==self.optimal_arm:
            self.rewards_list[arm].append(self.optimal_sampler.rvs(1))
        else:
            self.rewards_list[arm].append(self.dagger_sampler.rvs(1))          
    
    #def get_regret(self):
    #    total_reward = self.rewards_list[0] + self.rewards_list[1]
    #    regret = (self.base_p +self.delta) - sum(total_reward)/self.n
    #    return max(regret[0],0)            


class ImpUCB(Sampler):
    """
    This class implement improved-ucb algorithm 
    """
    def __init__(self,T,delta,optimal_arm,dist_type):
        super(ImpUCB, self).__init__(T,delta,optimal_arm,dist_type)
        self.t = 0
        self.r = [0,0]
        self.num_batches = 1
        self.app_delta = 1
        self.best_arm = None 
    
    def get_avg_reward_and_bs(self):
        """
        This function will return a averge reward and the confidense intervals. 
        """
        results = []
        bs = math.sqrt(math.log(self.T*self.app_delta**2)/self.t)
        for arm in range(2):
            ms = sum(self.rewards_list[arm])/len(self.rewards_list[arm])
            results.append([ms,bs])
        
        return results
    
    def set_best_arm(self):
        if self.best_arm is None:
            r_bs_arm1,r_bs_arm2 = self.get_avg_reward_and_bs()
            up_conf_arm1 = r_bs_arm1[0]+r_bs_arm1[1]
            low_conf_arm1 = r_bs_arm1[0]-r_bs_arm1[1]
            up_conf_arm2 = r_bs_arm2[0]+r_bs_arm2[1]
            low_conf_arm2 = r_bs_arm2[0]-r_bs_arm2[1]
            if low_conf_arm1 > up_conf_arm2:
                self.best_arm = 0
            elif low_conf_arm2 > up_conf_arm1:
                self.best_arm = 1
            else:  
                self.best_arm = None 
        
    def sample_reward(self):
        """
        This function will return a stochastic reward
        """ 
        batch_size = math.ceil(math.log(self.T*self.app_delta**2)/self.app_delta**2)
        if self.best_arm is not None:
            #print(self.T, self.num_batches)
            if self.best_arm == self.optimal_arm:
                self.rewards_list[self.optimal_arm] += self.optimal_sampler.rvs(2*batch_size).tolist()
            else:
                self.rewards_list[abs(self.optimal_arm-1)] += self.optimal_sampler.rvs(2*batch_size).tolist()    
        else:    
            self.rewards_list[self.optimal_arm] += self.optimal_sampler.rvs(batch_size).tolist()
            self.rewards_list[abs(self.optimal_arm-1)] += self.dagger_sampler.rvs(batch_size).tolist()          
            self.num_batches+=1
        self.t += batch_size*2
        
class ImpUCB_GTB(ImpUCB):
    """
    This class implement improved-ucb with go to broke for m>M-1 
    """
    def __init__(self,T,M,delta,optimal_arm,dist_type):
        super(ImpUCB_GTB, self).__init__(T,delta,optimal_arm,dist_type)
        self.M = M
    
    def set_best_arm(self):
        if self.best_arm is None:
            r_bs_arm1,r_bs_arm2 = self.get_avg_reward_and_bs()
            up_conf_arm1 = r_bs_arm1[0]+r_bs_arm1[1]
            low_conf_arm1 = r_bs_arm1[0]-r_bs_arm1[1]
            up_conf_arm2 = r_bs_arm2[0]+r_bs_arm2[1]
            low_conf_arm2 = r_bs_arm2[0]-r_bs_arm2[1]
            if low_conf_arm1 > up_conf_arm2:
                self.best_arm = 0
            elif low_conf_arm2 > up_conf_arm1:
                self.best_arm = 1
            else: 
                if self.num_batches==self.M-1:
                    self.best_arm = 0 if r_bs_arm1[0] > r_bs_arm2[0] else 1
                else:    
                    self.best_arm = None 
        

class BAE(ImpUCB):
    """
    This class implement Batched Arm Elimination algorithm from Esfandiari et al 2020
    """
    def __init__(self,T,M,delta,optimal_arm,dist_type):
        super(BAE, self).__init__(T,delta,optimal_arm,dist_type)
        self.M = M
        self.q = self.round2(T**(1/M))

    def get_avg_reward_and_bs(self):
        """
        This function will return a averge reward and the confidense intervals. 
        """
        results = []
        bs = math.sqrt(4*math.log(4*self.T*self.M)/self.t)
        for arm in range(2):
            ms = sum(self.rewards_list[arm])/len(self.rewards_list[arm])
            results.append([ms,bs])
        
        return results
    
    def sample_reward(self):
        """
        This function will return a stochastic reward
        """ 
        batch_size = self.q *2
        if self.best_arm is not None:
            #print(self.T, self.num_batches)
            if self.best_arm == self.optimal_arm:
                self.rewards_list[self.optimal_arm] += self.optimal_sampler.rvs(2*batch_size).tolist()
            else:
                self.rewards_list[abs(self.optimal_arm-1)] += self.optimal_sampler.rvs(2*batch_size).tolist()    
        else:    
            self.rewards_list[self.optimal_arm] += self.optimal_sampler.rvs(batch_size).tolist()
            self.rewards_list[abs(self.optimal_arm-1)] += self.dagger_sampler.rvs(batch_size).tolist()          
            self.num_batches+=1
        self.t += batch_size*2

    def set_best_arm(self):
        if self.best_arm is None:
            r_bs_arm1,r_bs_arm2 = self.get_avg_reward_and_bs()
            mean_arm1 = r_bs_arm1[0]
            low_conf_arm1 = r_bs_arm1[0]-r_bs_arm1[1]
            mean_arm2 = r_bs_arm2[0]
            low_conf_arm2 = r_bs_arm2[0]-r_bs_arm2[1]
            if low_conf_arm1 > mean_arm2:
                self.best_arm = 0
            elif low_conf_arm2 > mean_arm1:
                self.best_arm = 1
            else: 
                if self.num_batches==self.M-1:
                    self.best_arm = 0 if r_bs_arm1[0] > r_bs_arm2[0] else 1
                else:    
                    self.best_arm = None 


class PredTau(Sampler):
    """
    This class implement ucb2
    """
    def __init__(self,T,M,delta,optimal_arm,dist_type):
        super(PredTau, self).__init__(T,delta,optimal_arm,dist_type)
        self.r = [0,0]
        self.num_batches = 0
        self.app_delta = []
        self.best_arm = None 
        self.m=0
        self.M=M
        self.u0=self.round2(T**(1/(2-2**(1-M)))*math.log(T**(1/(2**M-1)))**(0.25-0.75/(2**M-1)))
    
    def reset(self):
        self.rewards_list=[[],[]]
        self.t=0
    def round2(self,x):
        return min(math.floor(x/2)*2,self.T)

    def approx_tau(self):
        r_bs_arm1,r_bs_arm2 = self.get_avg_reward_and_bs()
        std_avg = (r_bs_arm1[0]/r_bs_arm1[2]+r_bs_arm2[0]/r_bs_arm2[2])/2 > 0.5
        #delta = abs(r_bs_arm1[0]-r_bs_arm2[0])+1e-3-std_avg*math.log(math.sqrt(self.T/self.t)) #1e-3 #+(r_bs_arm1[1]+r_bs_arm2[1])/(2*self.T)
        
        if abs(r_bs_arm1[0]-r_bs_arm2[0])>r_bs_arm2[1] or std_avg:
            delta = abs(r_bs_arm1[0]-r_bs_arm2[0])+1e-3
        else:
            delta = (abs(r_bs_arm1[0]-r_bs_arm2[0])+1e-3)/(self.M-self.num_batches)
        #print(std_avg,abs(r_bs_arm1[0]-r_bs_arm2[0]),r_bs_arm2[1])            
        self.app_delta.append(delta)
        b=self.T #max(256/delta**2*math.log(self.T*delta**2/128),self.t+2)
        for tau in range(1,int(b)):
            if delta>math.sqrt(math.log(2*self.T/tau)/tau):
                break              
        tau = min(tau,float(self.T*(self.num_batches+1))/self.M)
        print(tau)
          
        return tau    
    
    def set_best_arm(self):
        r_bs_arm1,r_bs_arm2 = self.get_avg_reward_and_bs()
        up_conf_arm1 = r_bs_arm1[0]+r_bs_arm1[1]
        low_conf_arm1 = r_bs_arm1[0]-r_bs_arm1[1]
        up_conf_arm2 = r_bs_arm2[0]+r_bs_arm2[1]
        low_conf_arm2 = r_bs_arm2[0]-r_bs_arm2[1]
        if low_conf_arm1 > up_conf_arm2:
            self.best_arm = 0
        elif low_conf_arm2 > up_conf_arm1:
            self.best_arm = 1
        else:
            if self.m==(self.M-1):
                self.best_arm = 0 if r_bs_arm1[0] > r_bs_arm2[0] else 1
            else:    
                self.best_arm = None 
               
    def sample_reward(self):
        """
        This function will return a stochastic reward
        :param is_optimal: is the arm the optimal arm, if so p=0.5+delta else p=0.5
        :param delta: parameter for the optimal arm reward probability
        :return: the reward
        """ 
        self.m+=1
        if self.best_arm is not None:
            batch_size=int(self.T-self.t)
            arm=self.best_arm
            if arm==self.optimal_arm:
                self.rewards_list[arm]+=self.optimal_sampler.rvs(batch_size).tolist()
            else:    
                self.rewards_list[arm]+=self.dagger_sampler.rvs(batch_size).tolist()            
        else:    
            if self.m>1:    
                tau = self.approx_tau()
                # if we predict tau is smalller it means delta is larger thus we caould have made our decsion already
                if tau<self.t: 
                    tau=self.t+2    

            batch_size=self.round2(self.u0 if self.m==1 else self.round2((tau-self.t)/(self.M-self.m)))
            self.rewards_list[self.optimal_arm]+=self.optimal_sampler.rvs(batch_size//2).tolist()
            self.rewards_list[abs(self.optimal_arm-1)]+=self.dagger_sampler.rvs(batch_size//2).tolist()
        self.t+=batch_size
    def get_avg_reward_and_bs(self):
        """
        This function will return a averge reward and the confidense intervals. 
        """
        results = []
        for arm in range(2):
            s = len(self.rewards_list[arm])
            bs = 2*math.sqrt(2*math.log(self.T/s)/s)
            ms = sum(self.rewards_list[arm])/s
            std = np.array(self.rewards_list[arm]).std()
            results.append([ms,bs,std])
        return results    