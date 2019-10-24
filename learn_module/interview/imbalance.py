import numpy as np

import math
import os

import argparse
import shutil

import time

    
def balanceVOC_greedy(A,std_var_thresh):
    np_A_t = np.array(A).transpose()
    np_solution = np.array([[1]]*len(A))
    np_obj_count = np.matmul(np_A_t,np_solution)
    count = 0
    while(math.sqrt(np_obj_count.var())>std_var_thresh):
        label_score = (np_obj_count - np_obj_count.sum()/np_obj_count.size)/np_obj_count.var()
        sample_score = np.matmul(label_score.transpose(),np_A_t).transpose()
        sample_score[np_solution==0] = 0
        max_id = sample_score.argmax()
        np_solution[max_id] = 0
        np_obj_count = np.matmul(np_A_t,np_solution)
        count += 1
        print(count)
    return np_solution.tolist()

def balanceVOC_GA(A,m,iter):
    np_A_t = np.array(A).transpose()
    m_np_solution = np.random.randint(0,2,(len(A),len(np_A_t)))
    m_np_solution[:,-1] = 1
    np_obj_count = np.matmul(np_A_t,m_np_solution)
    num_min = np_obj_count[:,-1].min()
    best_solution = m_np_solution[:,-1].copy()
    best_solution_scores = -best_solution.std()-abs(num_min - best_solution.mean())
    m,n = m_np_solution.shape
    variate_rate = 0.003
    for i in range(iter):
        fathers = m_np_solution.copy()
        #Assessment
        counts = np.matmul(np_A_t, fathers)
        scores = np.mat([-counts[:,i].std()-abs(num_min - counts[:,i].mean()) for i in range(counts.shape[1])])
        #best
        this_best_index = np.argmax(scores)
        this_best_solution = fathers[:,this_best_index].copy()
        this_best_solution_scores = scores[this_best_index]
        print(num_min,this_best_solution.mean(),this_best_solution.std())
        if(this_best_solution_scores>best_solution_scores):
            best_solution = this_best_solution
            best_solution_scores = this_best_solution_scores
        exp_scores = np.exp(scores)
        sum_exp_scores = exp_scores.sum()
        probability = exp_scores/sum_exp_scores#softmax
        cum_prob = probability.cumsum()
        #select and breed
        select_inds = np.random.random(n*2)
        _be_selected = cum_prob.tile((n*2,1)) - select_inds
        _be_selected[_be_selected>0] = 1
        inds = _be_selected.argmax(1).reshape(n,2)
        for i in range(n):
            cross_site = int(np.random.random()*m)
            i1 = inds[i,0]
            i2 = inds[i,1]
            m_np_solution[:cross_site+1,i] = fathers[:,i1]
            m_np_solution[cross_site+1:,i] = fathers[:,i2]
        #variation
        variate_site = np.random.random(m_np_solution.shape)<variate_rate
        m_np_solution[variate_site] = (m_np_solution[variate_site]+1)%2
        
    return best_solution


if __name__=='__main__':
    with open('./imbalance.npy','rb') as fp:
        A = np.load(fp)
    
    #begin = datetime.datetime.now()
    #A,sample_name,label_dict,others_dict = countLabel(args.dir)
    ##A,sample_name,label_dict,others_dict = countLabel(r'G:\fusion_2018-08-20')
    #end = datetime.datetime.now()
    #print('time for statistics = '+str((end - begin).total_seconds()))
    ##input()

    begin = time.time()
    solution = balanceVOC_greedy(A,300)


    print('----Before Balance----')
    npA = np.array(A)
    count = npA.sum(0)
    print('       least: {0:>6}'.format(count.min()))
    print('        most: {0:>6}'.format(count.max()))
    print('    non-zero: {0:>6}'.format(np.count_nonzero(count)))
    print('         all: {0:>6}'.format(len(count)))
    print('image number: {0:>6}'.format(npA.shape[0]))
    #for i in range(len(count)):
    #    print('{0:>4}: {1}'.format(i,count[i]))
    print('\n')
    print('----After Balance----') 
    new_count = np.matmul(npA.transpose(),np.array(solution))
    print('       least: {0:>6}'.format(new_count.min()))
    print('        most: {0:>6}'.format(new_count.max()))
    print('    non-zero: {0:>6}'.format(np.count_nonzero(new_count)))
    print('         all: {0:>6}'.format(len(new_count)))
    print('image number: {0:>6}'.format(solution.count([1])))
    #for i in range(len(new_count)):
    #    print('{0:>4}: {1}'.format(i,new_count[i]))

    end =  time.time()
    print('time for balance = {0}s'.format(end - begin))
