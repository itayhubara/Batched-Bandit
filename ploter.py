import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

grid_types = ['optimal_grid','arithmetic_grid','geometric_grid','minimax_grid','minmax_grid_table1']
T_list = [1e3,5e3,1e4,2e4,3e4,5e4,1e5,1.5e5,2e5,2.5e5]

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--algs', default=['grids','ucb2','improved_ucb','improved_ucb_gtb','pred_tau','batch_elimination'], type=int, nargs='+',
                    help='agorithms types')
parser.add_argument('--dist_types', default=['bernoulli','norm','poisson','student_t'], type=str, nargs='+',
                    help='distribution types')
parser.add_argument('--sub_plot', dest='sub_plot', action='store_true', default=False,
                    help='sub_plot all distributions')                   




def main():
    args = parser.parse_args()
    if not args.sub_plot:   
        legend=[]
        for d in args.dist_types:
            for a in args.algs:
                res = np.load(args.results_dir+'/results_'+a+'_'+d+'.npy')
                if a == 'grids':
                    for gt_ind,gt in enumerate(grid_types):
                        plt.plot(T_list,res[:,gt_ind])
                        legend.append(grid_types[gt_ind])
                else:
                    plt.plot(T_list,res)
                    legend.append(a)
            plt.xlabel('T')
            plt.legend(legend)
            plt.ylabel('average regret per T')
            plt.title(d.upper())
            plt.show()
            import pdb; pdb.set_trace()
    else:
        num_plots =  len(args.dist_types)
        fig, axs = plt.subplots(num_plots//2+num_plots%2, num_plots//2)
        
        for d_ind,d in enumerate(args.dist_types):
            legend=[]
            for a in args.algs:
                res = np.load(args.results_dir+'/results_'+a+'_'+d+'.npy')
                if a == 'grids':
                    for gt_ind,gt in enumerate(grid_types):
                        axs[d_ind//2,d_ind%2].plot(T_list,res[:,gt_ind])
                        legend.append(grid_types[gt_ind])
                else:
                    axs[d_ind//2,d_ind%2].plot(T_list,res)
                    legend.append(a)
        #    import pdb; pdb.set_trace()        
            axs[d_ind//2,d_ind%2].set_xlabel('T')
            axs[d_ind//2,d_ind%2].legend(legend)
            axs[d_ind//2,d_ind%2].set_ylabel('average regret per T')
            axs[d_ind//2,d_ind%2].set_title(d.upper())
        fig.show()    
        import pdb; pdb.set_trace()  
if __name__ == '__main__':
    main()