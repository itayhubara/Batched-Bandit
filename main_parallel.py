import argparse
from grid_policies import *
from sampler import Sampler
from algorithms import UCB2,ImpUCB,ImpUCB_GTB,PredTau,BAE
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import os

dist_types=['bernoulli','norm','poisson','student_t']
grids = [optimal_grid,arithmetic_grid,geometric_grid,minimax_grid,minmax_grid_paper]
T_list = [1e3,5e3,1e4,2e4,3e4,5e4,1e5,1.5e5,2e5,2.5e5]


algs = ['grids','improved_ucb','improved_ucb_gtb','ucb2','pred_tau','batch_elimination']
parser = argparse.ArgumentParser(description='Batched Bandit Paper')

parser.add_argument('--alg', '-a', metavar='ALG', default='grids',
                    choices=algs,
                    help=' algorithm type: ' +
                    ' | '.join(algs) +
                    ' (default: grids)')
parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--delta', default=0.1,
                    help='delta between arms')
parser.add_argument('--num_repeat', default=100,
                    help='number of repetited experiment per T - for rubstness')   
parser.add_argument('--M', default=5,
                    help='number of batched for the grids and pred_tau alg')                                       
parser.add_argument('--optimal_arm', default=1,
                    help='optimal arm must be 0 or 1')  
parser.add_argument('--alpha', default=0.2,
                    help='alpha for ucb2 algorithm')                     
parser.add_argument('--distribution', '-d', metavar='DIST', default='bernoulli',
                    choices=dist_types,
                    help=' arms distribution type: ' +
                    ' | '.join(algs) +
                    ' (default: bernoulli)')                     

     
def paralle_exp(args,decision_points,T):
    sampler = Sampler(T,args.delta,args.optimal_arm,args.distribution)
    sampler.reset()
    best_arm = None
    arm=np.random.randint(2)
    for t in range(int(T)): 
            best_arm =  sampler.return_best_arm(decision_points,best_arm,t)
            arm = best_arm if best_arm else abs(arm-1)  
            sampler.sample_reward(arm)    
    return sampler.get_regret()

def paralle_ucb2(args,T):
    sampler = UCB2(args.alpha,args.delta,args.optimal_arm,args.distribution)
    sampler.select_arm_update()
    while sampler.t<T:
        arm,batch_size = sampler.select_arm_update()
        for i in range(batch_size):
            sampler.sample_reward(arm)  
    return sampler.get_regret()

def paralle_imp_ucb(args,T):
    sampler = ImpUCB(T,args.delta,args.optimal_arm,args.distribution)
    while sampler.t<T:
        sampler.sample_reward()
        sampler.set_best_arm()
        sampler.app_delta /= 2    
    return sampler.get_regret()

def paralle_imp_ucb_gtb(args,T):
    sampler = ImpUCB_GTB(T,args.M,args.delta,args.optimal_arm,args.distribution)
    while sampler.t<T:
        sampler.sample_reward()
        sampler.set_best_arm()
        sampler.app_delta /= 2    
    return sampler.get_regret()

def paralle_pred_tau(args,T):
    sampler = PredTau(T,args.M,args.delta,args.optimal_arm,args.distribution)
    while sampler.t<T:
        assert sampler.m<sampler.M
        sampler.sample_reward()
        sampler.set_best_arm()
    return sampler.get_regret()

def paralle_batch_elimination(args,T):
    sampler = BAE(T,args.M,args.delta,args.optimal_arm,args.distribution)
    while sampler.t<T:
        sampler.sample_reward()
        sampler.set_best_arm()  
    return sampler.get_regret()

def main():
    args = parser.parse_args()
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if args.alg=='grids':
        final_results = np.zeros([len(T_list),len(grids)])
        for tidx,T in enumerate(T_list):
            sampler = Sampler(T,args.delta,args.optimal_arm,args.distribution)
            grid_results=[]
            for gidx,grid in enumerate(grids):
                decision_points = grid(T,args.M) if gidx !=0 else grid(T,args.delta)
                print('grid decision points t_m: ', decision_points)
                regret = Parallel(n_jobs=40)(delayed(paralle_exp)(args,decision_points,T) for i in range(args.num_repeat))
                grid_results.append(sum(regret)/100)
            final_results[tidx,:]=grid_results        
            print(final_results)    
    if args.alg=='ucb2':
        final_results = np.zeros([len(T_list)])
        for tidx,T in enumerate(T_list):
            paralle_ucb2(args,T)
            regret = Parallel(n_jobs=40)(delayed(paralle_ucb2)(args,T) for i in range(args.num_repeat))
            final_results[tidx]=sum(regret)/100  
            print(final_results) 
    if args.alg=='improved_ucb':
        final_results = np.zeros([len(T_list)])
        for tidx,T in enumerate(T_list):
            paralle_imp_ucb(args,T)
            regret = Parallel(n_jobs=40)(delayed(paralle_imp_ucb)(args,T) for i in range(args.num_repeat))
            final_results[tidx]=sum(regret)/100  
        print(final_results) 
    if args.alg=='improved_ucb_gtb':
        final_results = np.zeros([len(T_list)])
        for tidx,T in enumerate(T_list):
            paralle_imp_ucb(args,T)
            regret = Parallel(n_jobs=40)(delayed(paralle_imp_ucb_gtb)(args,T) for i in range(args.num_repeat))
            final_results[tidx]=sum(regret)/100  
        print(final_results)         
    if args.alg=='pred_tau':
        final_results = np.zeros([len(T_list)])
        for tidx,T in enumerate(T_list):
            #print(paralle_pred_tau(args,T))
            #import pdb; pdb.set_trace()
            regret = Parallel(n_jobs=40)(delayed(paralle_pred_tau)(args,T) for i in range(args.num_repeat))
            final_results[tidx]=sum(regret)/100  
        print(final_results)
    if args.alg=='batch_elimination':
        final_results = np.zeros([len(T_list)])
        for tidx,T in enumerate(T_list):
            #print(paralle_batch_elimination(args,T))
            #import pdb; pdb.set_trace()
            regret = Parallel(n_jobs=40)(delayed(paralle_batch_elimination)(args,T) for i in range(args.num_repeat))
            final_results[tidx]=sum(regret)/100  
        print(final_results)                
    np.save('./results/results_'+args.alg+'_'+args.distribution,final_results)    

if __name__ == '__main__':
    main()    