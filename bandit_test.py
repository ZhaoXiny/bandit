import numpy as np
import pymc
from matplotlib import pylab as plt
#from mpltools import style # uncomment for prettier plots
#style.use(['ggplot'])

'''
function definitions
'''

def generate_bernoulli_bandit_data(num_samples,K):
    CTRs_that_generated_data = np.tile(np.random.rand(K),(num_samples,1))  
    true_rewards = np.random.rand(num_samples,K) < CTRs_that_generated_data
    return true_rewards,CTRs_that_generated_data

# totally random
def random(estimated_beta_params):
    return np.random.randint(0,len(estimated_beta_params))

# the naive algorithm
def naive(estimated_beta_params,number_to_explore=100):
    totals = estimated_beta_params.sum(1) 
    if np.any(totals < number_to_explore): 
        least_explored = np.argmin(totals) 
        return least_explored
    else: 
        successes = estimated_beta_params[:,0] 
        estimated_means = successes/totals 
        best_mean = np.argmax(estimated_means) # the best
        return best_mean

# the epsilon greedy algorithm
def epsilon_greedy(estimated_beta_params,epsilon=0.01):
    totals = estimated_beta_params.sum(1) # The number of experiments per arm
    successes = estimated_beta_params[:,0] # successes
    estimated_means = successes/totals # the current means
    best_mean = np.argmax(estimated_means) # the best mean
    be_exporatory = np.random.rand() < epsilon # should we explore?
    if be_exporatory: # totally random, excluding the best_mean
        other_choice = np.random.randint(0,len(estimated_beta_params))
        while other_choice == best_mean:
            other_choice = np.random.randint(0,len(estimated_beta_params))
        return other_choice
    else: # take the best mean
        return best_mean

# the UCB algorithm using 
def UCB(estimated_beta_params):
    t = float(estimated_beta_params.sum()) # total number of rounds 
    totals = estimated_beta_params.sum(1)  #The number of experiments per arm
    successes = estimated_beta_params[:,0]
    estimated_means = successes/totals # earnings mean
    estimated_variances = estimated_means - estimated_means**2    
    UCB = estimated_means + np.sqrt( np.minimum( estimated_variances + np.sqrt(2*np.log(t)/totals), 0.25 ) * np.log(t)/totals )
    return np.argmax(UCB)

# the Thompson sampling algorithm
def Thompson_sampling(estimated_beta_params):
    totals = estimated_beta_params.sum(1)  #The number of experiments per arm
    successes = estimated_beta_params[:,0]
    return np.argmax(pymc.rbeta(1 + successes, 1 + totals - successes))    

# the bandit algorithm
def run_bandit_dynamic_alg(true_rewards,CTRs_that_generated_data,choice_func):
    num_samples,K = true_rewards.shape 
    #
    prior_a = 1. # 
    prior_b = 1. #
    estimated_beta_params = np.zeros((K,2)) #记录没一个臂的成功和失败次数
    estimated_beta_params[:,0] += prior_a # 
    estimated_beta_params[:,1] += prior_b
    regret = np.zeros(num_samples)

    for i in range(0,num_samples):
        
        this_choice = choice_func(estimated_beta_params)

        if true_rewards[i,this_choice] == 1:
            update_ind = 0
        else:
            update_ind = 1
            
        estimated_beta_params[this_choice,update_ind] += 1
        
        #第i次实验的遗憾 
        regret[i] = np.max(CTRs_that_generated_data[i,:]) - CTRs_that_generated_data[i,this_choice]

    cum_regret = np.cumsum(regret)

    return cum_regret

'''
main code
'''
#
num_samples = 10000
K = 5 # 
number_experiments = 50 #
#记录5种方法的累计遗憾
regret_accumulator = np.zeros((num_samples,5))
for i in range(number_experiments):
    print "Running experiment:", i+1
    true_rewards,CTRs_that_generated_data = generate_bernoulli_bandit_data(num_samples,K)
    regret_accumulator[:,0] += run_bandit_dynamic_alg(true_rewards,CTRs_that_generated_data,random)#
    regret_accumulator[:,1] += run_bandit_dynamic_alg(true_rewards,CTRs_that_generated_data,naive)
    regret_accumulator[:,2] += run_bandit_dynamic_alg(true_rewards,CTRs_that_generated_data,epsilon_greedy)
    regret_accumulator[:,3] += run_bandit_dynamic_alg(true_rewards,CTRs_that_generated_data,UCB)
    regret_accumulator[:,4] += run_bandit_dynamic_alg(true_rewards,CTRs_that_generated_data,Thompson_sampling)
#  
plt.semilogy(regret_accumulator/number_experiments)
plt.title('Simulated Bandit Performance for K = 5')
plt.ylabel('Cumulative Expected Regret')
plt.xlabel('Round Index')
plt.legend(('Random','Naive','Epsilon-Greedy','UCB','Thompson sampling'),loc='lower right')
plt.show()
