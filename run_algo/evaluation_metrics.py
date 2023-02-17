import numpy as np
from run_optimizer import get_opt_f, get_abs_opt_f


# cosine similarity metric
def cosine_metric(w, true_w):
    
    cosine_similarity = np.dot(w, true_w)/(np.linalg.norm(w)*np.linalg.norm(true_w))
    
    return cosine_similarity

# simple regret metric
def simple_regret(predefined_features, w, true_w):
    # true reward
    opt_feature = get_opt_f(predefined_features, w)
    true_opt_feature = get_opt_f(predefined_features, true_w)
    
    true_reward = np.dot(true_w, true_opt_feature)
    simple_reward = np.dot(true_w, opt_feature) # simple reward
    
    return simple_reward


# cumulative regret    
def regret(predefined_features, A, true_w):
    # regret
    opt_feature = get_abs_opt_f(predefined_features, true_w)
    
    opt_r = np.abs(np.dot(true_w, opt_feature))
    current_r = np.abs(np.dot(true_w, A.T))
    
    
    regret = opt_r-current_r
    return np.sum(regret)