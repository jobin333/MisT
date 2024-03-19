import torch
import os 
from torch.utils.data import DataLoader
from mvit.logging_utils.logger import logger
import numpy as np

module_logger = logger.getChild(__name__)

class DatasetManager():
  def __init__(self, feature_path, discount=0.9):
    indices = range(1, 81)
    self.feature_path = feature_path
    self.discount = discount
    self.data = self.load_files(indices)

  def __len__(self):
    return len(self.data)

  def get_dataloader(self, idx):
      return self.data[idx+1]

  def load_files(self, indices):
    file_paths = [os.path.join(self.feature_path, f'tensors_{i}.pt') for i in indices]
    files = [torch.load(path) for path in file_paths ]
    data = [self.add_cumsum_info(file) for file in files]
    return data

  def concat_file_data(self, files_data):
    data_dict = {}
    stacked_dict = {}

    for file_dict in files_data:
      for key, value in file_dict.items():
        if key not in data_dict:
          data_dict[key] = [value]
        else:
          data_dict[key] += [value]

    for key, value in data_dict.items():
      stacked_dict[key] = torch.cat(data_dict[key])
    return stacked_dict


  def add_cumsum_info(self, feature_data):
    feature, phase, tool = zip(*feature_data)
    length = len(feature)
    feature = torch.stack(feature)
    phase = torch.stack(phase)
    tool = torch.stack(tool)
    onehot_phase = torch.nn.functional.one_hot(phase, num_classes=7)
    cumsum_phase = torch.cumsum(onehot_phase, 0)
    cumsum_tool = torch.cumsum(tool, 0)
    d_cumsum_phase = self.discounted_cumsum(onehot_phase)
    d_cumsum_tool = self.discounted_cumsum(tool)
    data = {'feature':feature, 'phase':phase, 'tool':tool,
            'cumsum_phase':cumsum_phase, 'cumsum_tool':cumsum_tool,
               'd_cumsum_phase':d_cumsum_phase, 'd_cumsum_tool':d_cumsum_tool,
            'onehot_phase':onehot_phase}
    return data

  def discounted_cumsum(self, series):
    cumsum = [series[0] * 0 ]
    for item in series:
      data = item + cumsum[-1] * self.discount
      cumsum.append(data)
    cumsum.pop()
    return torch.stack(cumsum)
