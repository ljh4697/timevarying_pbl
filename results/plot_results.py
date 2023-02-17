import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_bench_results, plot_cosine_metric, plot_simple_regret, plot_cumulative_regret




driver_opt_params ={
    'delta':0.7,
    'alpha':0.0002,
    'gamma':0.952,
    'lambda':0.1
}

tosser_opt_params = {
    'delta':0.7,
    'alpha':0.0009,
    'gamma':0.954,
    'lambda':0.2
}

avoid_opt_params = {
    'delta':0.7,
    'alpha':0.0002,
    'gamma':0.95,
    'lambda':0.42
}
v_tosser_opt_params = {
    'delta':0.7,
    'alpha':0.0009,
    'gamma':0.954,
    'lambda':0.19,
    'seed':4
}


task = 'driver'

# delta = 0.7
# alpha = 0.25 # 0.0005 for avoid 0.0002 for drive
# gamma = 0.935
# lamb = 0.8

delta = globals()[task+'_opt_params']['delta']
alpha =globals()[task+'_opt_params']['alpha']
gamma = globals()[task+'_opt_params']['gamma']
lamb = globals()[task+'_opt_params']['lambda']

DPB_cosine = []
DPB_simple_regret = []
DPB_cumulative_regret = []
DPB_opt_simple_reward = []

DPB2_cosine = []
DPB2_simple_regret = []
DPB2_cumulative_regret = []
DPB2_opt_simple_reward = []

for i in range(1, 11):
    
    #tosser, avoid
    DPB_result = np.load(task + '/DPB/' + 'n{:}-iter400-DPB-delta{:.2f}-alpha{:.4f}-gamma{:.3f}-lambda{:.2f}-seed{:d}.npy'.format(task, delta, alpha, gamma, lamb, i))
    
    #driver
    #DPB_result = np.load(task + '/DPB/' + '{:}-iter400-DPB-delta{:.2f}-alpha{:.4f}-gamma{:.2f}-seed{:d}.npy'.format(task, delta, alpha, gamma, i))

    DPB_cosine.append(DPB_result['eval_cosine'])
    DPB_simple_regret.append(DPB_result['opt_simple_reward']-DPB_result['eval_simple_regret'])
    DPB_opt_simple_reward.append(DPB_result['opt_simple_reward'])
    DPB_cumulative_regret.append(DPB_result['eval_cumulative_regret'])
    
        
DPB_cosine_evaluation = np.mean(DPB_cosine, axis=0)
DPB_cosine_evaluation_std = np.std(DPB_cosine, axis=0)*0.5

DPB_simple_regret_evaluation = np.mean(DPB_simple_regret, axis=0)
DPB_simple_regret_evaluation_std = np.std(DPB_simple_regret, axis=0)*0.5

opt_simple_reward = np.mean(DPB_opt_simple_reward, axis=0)
opt_simple_reward_std = np.std(DPB_opt_simple_reward, axis=0)

DPB_cumulative_regret_evaluation = np.mean(DPB_cumulative_regret, axis=0)
DPB_cumulative_regret_evaluation_std = np.std(DPB_cumulative_regret, axis=0)


dpb2_driver_opt_params ={
    'delta':0.7,
    'alpha':1.5,
    'gamma':0.945,
    'lambda':0.3
}

dpb2_tosser_opt_params = {
    'delta':0.7,
    'alpha':0.0004,  # 8 point
    'gamma':0.95,
    'lambda':0.5
}

dpb2_avoid_opt_params = {
    'delta':0.7,
    'alpha':0.0009,
    'gamma':0.95, # 94 point
    'lambda':0.5
}



delta2 = globals()['dpb2_'+task+'_opt_params']['delta']
alpha2 =globals()['dpb2_'+task+'_opt_params']['alpha']
gamma2 = globals()['dpb2_'+task+'_opt_params']['gamma']
lamb2 = globals()['dpb2_'+task+'_opt_params']['lambda']

for i in range(1, 11):
    
    
    DPB2_result = np.load(task + '/DPB2/' + '{:}-iter400-DPB2-delta{:.2f}-alpha{:.4f}-gamma{:.3f}-lambda{:.2f}-seed{:d}.npy'.format(task, delta2, alpha2, gamma2, lamb2, i))

    DPB2_cosine.append(DPB2_result['eval_cosine'])
    DPB2_simple_regret.append(opt_simple_reward-DPB2_result['eval_simple_regret'])
    DPB2_cumulative_regret.append(DPB2_result['eval_cumulative_regret'])
    
        
DPB2_cosine_evaluation = np.mean(DPB2_cosine, axis=0)
DPB2_cosine_evaluation_std = np.std(DPB2_cosine, axis=0)*0.5

DPB2_simple_regret_evaluation = np.mean(DPB2_simple_regret, axis=0)
DPB2_simple_regret_evaluation_std = np.std(DPB2_simple_regret, axis=0)*0.5



DPB2_cumulative_regret_evaluation = np.mean(DPB2_cumulative_regret, axis=0)
DPB2_cumulative_regret_evaluation_std = np.std(DPB2_cumulative_regret, axis=0)










# bench marking algorithms' results
(BA_greedy_cosine_evaluation, BA_greedy_cosine_evaluation_std,
 BA_greedy_simple_regret_evaluation, BA_greedy_simple_regret_evaluation_std,
 BA_greedy_cumulative_regret_evaluation, BA_greedy_cumulative_regret_evaluation_std) = get_bench_results(task, 'greedy', 10, opt_simple_reward)

(BA_medoids_cosine_evaluation, BA_medoids_cosine_evaluation_std,
 BA_medoids_simple_regret_evaluation, BA_medoids_simple_regret_evaluation_std,
 BA_medoids_cumulative_regret_evaluation, BA_medoids_cumulative_regret_evaluation_std) = get_bench_results(task, 'medoids', 10, opt_simple_reward)

(BA_dpp_cosine_evaluation, BA_dpp_cosine_evaluation_std,
 BA_dpp_simple_regret_evaluation, BA_dpp_simple_regret_evaluation_std,
 BA_dpp_cumulative_regret_evaluation, BA_dpp_cumulative_regret_evaluation_std) = get_bench_results(task, 'dpp', 10, opt_simple_reward)

(random_cosine_evaluation, random_cosine_evaluation_std,
 random_simple_regret_evaluation, random_simple_regret_evaluation_std,
 random_cumulative_regret_evaluation, random_cumulative_regret_evaluation_std) = get_bench_results(task, 'random', 10, opt_simple_reward)






''''''
# plot_cosine_metric(DPB_cosine_evaluation, DPB_cosine_evaluation_std,
#                    DPB2_cosine_evaluation, DPB2_cosine_evaluation_std,
#                    BA_greedy_cosine_evaluation, BA_greedy_cosine_evaluation_std,
#                    BA_medoids_cosine_evaluation, BA_medoids_cosine_evaluation_std,
#                    BA_dpp_cosine_evaluation, BA_dpp_cosine_evaluation_std,
#                    random_cosine_evaluation, random_cosine_evaluation_std, task=task)



# plot_simple_regret(opt_simple_reward, opt_simple_reward,
#                    DPB_simple_regret_evaluation, DPB_simple_regret_evaluation_std,
#                    BA_greedy_simple_regret_evaluation, BA_greedy_simple_regret_evaluation_std,
#                    BA_medoids_simple_regret_evaluation, BA_medoids_simple_regret_evaluation_std,
#                    BA_dpp_simple_regret_evaluation, BA_dpp_simple_regret_evaluation_std,
#                    random_simple_regret_evaluation, random_simple_regret_evaluation_std, task=task)

# plot_cumulative_regret(DPB_cumulative_regret_evaluation, DPB_cumulative_regret_evaluation_std,
#                    BA_greedy_cumulative_regret_evaluation, BA_greedy_cumulative_regret_evaluation_std,
#                    BA_medoids_cumulative_regret_evaluation, BA_medoids_cumulative_regret_evaluation_std,
#                    BA_dpp_cumulative_regret_evaluation, BA_dpp_cumulative_regret_evaluation_std,
#                    random_cumulative_regret_evaluation, random_cumulative_regret_evaluation_std, task=task)











# #### subplot
fg = plt.figure(figsize=(15,4))
b = 10
cosine_metric = fg.add_subplot(131)
simple_regret_metric = fg.add_subplot(132)
cumulative_regret_metric = fg.add_subplot(133)



cosine_metric.plot(b*np.arange(len(DPB_cosine_evaluation)), DPB_cosine_evaluation, color='orange', label='DPB', alpha=0.8)
cosine_metric.plot(b*np.arange(len(DPB2_cosine_evaluation)), DPB2_cosine_evaluation, color='darkblue', label='DPB2', alpha=0.8)

cosine_metric.plot(b*np.arange(len(BA_greedy_cosine_evaluation)), BA_greedy_cosine_evaluation, color='red', label='greedy', alpha=0.4)
cosine_metric.plot(b*np.arange(len(BA_medoids_cosine_evaluation)), BA_medoids_cosine_evaluation, color='red', label='medoids', alpha=0.6)
cosine_metric.plot(b*np.arange(len(BA_dpp_cosine_evaluation)), BA_dpp_cosine_evaluation, color='red', label='dpp', alpha=0.8)
cosine_metric.plot(b*np.arange(len(random_cosine_evaluation)), random_cosine_evaluation, color='green', label='random', alpha=0.8)


cosine_metric.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
cosine_metric.axvline(x=200, color='gray', linestyle='--', alpha=0.7)
cosine_metric.set_ylabel('m')
cosine_metric.set_xlabel('N')
cosine_metric.set_title('cosine metric')
cosine_metric.set_ylim((-1, 1))
cosine_metric.legend()


simple_regret_metric.plot(b*np.arange(len(DPB_simple_regret_evaluation)), DPB_simple_regret_evaluation, color='orange', label='DPB', alpha=0.8)
simple_regret_metric.plot(b*np.arange(len(DPB2_simple_regret_evaluation)), DPB2_simple_regret_evaluation, color='darkblue', label='DPB2', alpha=0.8)
simple_regret_metric.plot(b*np.arange(len(BA_greedy_simple_regret_evaluation)), BA_greedy_simple_regret_evaluation, color='red', label='greedy', alpha=0.4)
simple_regret_metric.plot(b*np.arange(len(BA_medoids_simple_regret_evaluation)), BA_medoids_simple_regret_evaluation, color='red', label='medoids', alpha=0.6)
simple_regret_metric.plot(b*np.arange(len(BA_dpp_simple_regret_evaluation)), BA_dpp_simple_regret_evaluation, color='red', label='dpp', alpha=0.8)
simple_regret_metric.plot(b*np.arange(len(random_simple_regret_evaluation)), random_simple_regret_evaluation, color='green', label='random', alpha=0.8)

simple_regret_metric.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
simple_regret_metric.axvline(x=200, color='gray', linestyle='--', alpha=0.7)
simple_regret_metric.set_ylabel('m')
simple_regret_metric.set_xlabel('N')
simple_regret_metric.set_title('simple regret')
simple_regret_metric.legend()

cumulative_regret_metric.plot(b*np.arange(len(DPB_cumulative_regret_evaluation)), DPB_cumulative_regret_evaluation, color='orange', label='DPB', alpha=0.8)
cumulative_regret_metric.plot(b*np.arange(len(DPB2_cumulative_regret_evaluation)), DPB2_cumulative_regret_evaluation, color='darkblue', label='DPB2', alpha=0.8)
cumulative_regret_metric.plot(b*np.arange(len(BA_greedy_cumulative_regret_evaluation)), BA_greedy_cumulative_regret_evaluation, color='red', label='greedy', alpha=0.4)
cumulative_regret_metric.plot(b*np.arange(len(BA_medoids_cumulative_regret_evaluation)), BA_medoids_cumulative_regret_evaluation, color='red', label='medoids', alpha=0.6)
cumulative_regret_metric.plot(b*np.arange(len(BA_dpp_cumulative_regret_evaluation)), BA_dpp_cumulative_regret_evaluation, color='red', label='dpp', alpha=0.8)
cumulative_regret_metric.plot(b*np.arange(len(random_cumulative_regret_evaluation)), random_cumulative_regret_evaluation, color='green', label='random', alpha=0.8)


cumulative_regret_metric.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
cumulative_regret_metric.axvline(x=200, color='gray', linestyle='--', alpha=0.7)
cumulative_regret_metric.set_ylabel('m')
cumulative_regret_metric.set_xlabel('N')
cumulative_regret_metric.set_title('cumulative regret')
cumulative_regret_metric.legend()

#plt.title(task + '/DPB/' + '{:}-iter400-DPB-delta{:.2f}-alpha{:.4f}-gamma{:.3f}-lambda{:.2f}-seed{:d}.npy'.format(task, delta, alpha, gamma, lamb, i))
plt.show()

