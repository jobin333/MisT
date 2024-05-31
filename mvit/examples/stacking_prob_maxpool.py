import torch

def generate_stack(x, stack_length, roll_count):
    if len(x.shape) == 3:  ## Already stacked
        return x
    stack_list = [x]
    for i in range(stack_length-1):
        shifted_x = torch.roll(stack_list[-1], roll_count, dims=0)
        shifted_x[:roll_count] = 0.
        stack_list.append(shifted_x)
    stack = torch.stack(stack_list).permute(1,0,2)
    return stack

def prob_maxpool(data, maxpool_window_size):
    soft = torch.nn.Softmax(-1)
    stacked_data = generate_stack(data , maxpool_window_size, 1)
    max_values, _ = torch.max(stacked_data, dim=-1)
    indices = torch.argmax(max_values, -1)
    max_prob = torch.stack([stacked_data[i,j] for i,j in enumerate(indices)])