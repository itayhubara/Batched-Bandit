from scipy.stats import bernoulli, norm, poisson
from scipy.stats import t as student_t
import math

class Sampler:
    """
    This class will help handle all of the means and variances of the arms rewards
    """
    def __init__(self, T, delta, optimal_arm, dist_type):
        self.optimal_arm = optimal_arm
        self.base_p =  0.5
        self.delta = delta
        self.rewards_list=[[],[]]
        self.t=0
        self.T = T 
        self.return_num_batches = False
            
        if dist_type == 'bernoulli':
            self.optimal_sampler = bernoulli(self.base_p + delta)
            self.dagger_sampler = bernoulli(self.base_p)
        elif dist_type == 'norm':
            self.optimal_sampler = norm(self.base_p + delta,1)
            self.dagger_sampler = norm(self.base_p,1)
        elif dist_type == 'poisson':
            self.optimal_sampler = poisson(self.base_p + delta)
            self.dagger_sampler = poisson(self.base_p)  
        elif dist_type == 'student_t':
            self.optimal_sampler = student_t(2, loc=self.base_p+delta)
            self.dagger_sampler = student_t(2,loc=self.base_p)                       

    def reset(self):
        self.rewards_list=[[],[]]
        self.t=0

    def round2(self,x):
        return min(math.floor(x/2)*2,self.T)
        
    def return_best_arm(self,decision_points,best_arm,t):
        if t in decision_points and best_arm is None:      
            r_bs_arm1,r_bs_arm2 = self.get_avg_reward_and_bs()
            up_conf_arm1 = r_bs_arm1[0]+r_bs_arm1[1]
            low_conf_arm1 = r_bs_arm1[0]-r_bs_arm1[1]
            up_conf_arm2 = r_bs_arm2[0]+r_bs_arm2[1]
            low_conf_arm2 = r_bs_arm2[0]-r_bs_arm2[1]
            if low_conf_arm1 > up_conf_arm2:
                best_arm = 0
            elif low_conf_arm2 > up_conf_arm1:
                best_arm = 1
            else:
                if t==decision_points[-2]:
                    best_arm = 0 if r_bs_arm1[0] > r_bs_arm2[0] else 1
                else:    
                    best_arm = None 
        return best_arm   
    
    def sample_reward(self,arm):
        """
        This function will return a stochastic reward
        :param is_optimal: is the arm the optimal arm, if so p=0.5+delta else p=0.5
        :param delta: parameter for the optimal arm reward probability
        :return: the reward
        """ 
        self.t+=1
        if arm==self.optimal_arm:
            self.rewards_list[arm].append(self.optimal_sampler.rvs(1))
        else:
            self.rewards_list[arm].append(self.dagger_sampler.rvs(1))    


    def get_avg_reward_and_bs(self):
        """
        This function will return a averge reward and the confidense intervals. 
        """
        results = []
        for arm in range(2):
            s = len(self.rewards_list[arm])
            bs = 2*math.sqrt(2*math.log(self.T/s)/s)
            ms = sum(self.rewards_list[arm])/s
            results.append([ms,bs])

        return results

    def get_regret(self):
        total_reward = self.rewards_list[0] + self.rewards_list[1]
        regret = (self.base_p +self.delta) - sum(total_reward)/self.t
        if isinstance(regret, list): regret=regret[0]
        if self.return_num_batches:
            return self.num_batches #max(regret,0)
        else:
            return max(regret,0)    
