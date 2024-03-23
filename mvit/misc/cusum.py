import torch

## function to implement mean adjusted cusum https://www.youtube.com/watch?v=bWhx80Xpsc4

def increase_cusum_loop(data, mean, slack):
  cusum = [0]
  for y in data:
    s = cusum[-1] + (y - mean) - slack
    s = torch.nn.functional.relu(s)
    cusum.append(s)
  cusum.pop(0)
  return torch.stack(cusum)

def decrease_cusum_loop(data, mean, slack):
  cusum = [0]
  for y in data:
    s = cusum[-1] - (y - mean) - slack
    s = torch.nn.functional.relu(s)
    cusum.append(s)
  cusum.pop(0)
  return torch.stack(cusum)

def view_list(data):
  for i in data:
    print(int(i.item()), end=' ')
  print()

def gen_cusum_mean_variance(pred, target):

  onehot = torch.nn.functional.one_hot(target)

  ## Mean
  product = pred * onehot
  mean = torch.sum(product, 0) / torch.sum(onehot, 0)

  # Variance
  pred_m_mean = pred - mean
  pred_m_mean_2 = pred_m_mean ** 2
  pred_m_mean_2 = pred_m_mean_2 * onehot
  variance = torch.sum(pred_m_mean_2, 0) / torch.sum(onehot, 0)

  return mean, variance


def gen_cusum(sequence, mean, slack):

  in_cusum = increase_cusum_loop(sequence, mean, slack)
  de_cusum = decrease_cusum_loop(sequence, mean, slack)

  return in_cusum, de_cusum

