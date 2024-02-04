'''
The phase transition sequence return the phase transition happen in the video data

Example:

[[6, 0, 2, 3, 4, 1, 5],
 [6, 0, 2, 3, 4, 1, 5],
 [6, 0],
 [0, 2, 3, 4, 1, 5],
 [6, 0, 2, 3, 4, 1, 5],
 [6, 0, 2, 3, 4, 1, 5],
 [6, 0, 2, 3, 4, 1, 5],
 [6, 0, 2, 3, 4],
 [4, 1, 5],
 [6, 0, 2, 3, 4, 1, 5],
 [6, 0, 2, 3, 4, 1, 5],
 [0, 2, 3, 4, 1, 5],
 [6, 0, 2, 3, 4, 1, 5],]
 
'''

def _get_single_transition_sequence(y):
    y = y.detach().cpu().numpy()
    yt = []
    old = y[0]
    yt.append(old)
    for new in y:
        if new != old:
            yt.append(new)
            old = new
    return yt
    
    
def get_phase_transition_sequence(dataset_manager):
    sequences = []
    for i in range(1, 81):
        ds = dataset_manager.get_dataloader(i)
        for x,y in ds:
            sequence = _get_single_transition_sequence(y)
            sequences.append(sequence)
    return sequences