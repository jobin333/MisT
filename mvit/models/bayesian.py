import torch

def get_trans_matrix(state_stay_factor, state_seq_factor, regularization_factor):
    other_trans_matrix  = [
        [0,0,1,0,0,0,0],
        [0,0,0,0,1,1,0],
        [0,0,0,1,0,0,0],
        [0,1,0,0,1,0,0],
        [0,1,0,0,0,1,0],
        [0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0],
    ]
    other_trans_matrix = torch.tensor(other_trans_matrix)
    self_trans_matrix = torch.eye(7) 

    trans_matrix = other_trans_matrix*state_seq_factor + self_trans_matrix*state_stay_factor + torch.ones((7,7))*regularization_factor
    return trans_matrix

def get_bayesian_prob(last_prediction, prob, trans_matrix):
    bayesian_prob = trans_matrix[last_prediction] * prob
    return bayesian_prob.argmax(-1)

def generate_bayesian_prediction(x, state_stay_factor=0.3, state_seq_factor=0.1, regularization_factor=1.):
    trans_matrix = get_trans_matrix(state_stay_factor, state_seq_factor, regularization_factor)
    trans_matrix = trans_matrix.to(x.device)
    last_prediction = x[0].argmax(-1)
    predictions = []
    for prob in x:
        last_prediction = get_bayesian_prob(last_prediction, prob, trans_matrix)
        predictions.append(last_prediction)
    predictions = torch.stack(predictions)
    return predictions