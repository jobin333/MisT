import torch
import os 
from torch.utils.data import DataLoader
from mvit.logging_utils.logger import logger

module_logger = logger.getChild(__name__)
class DictDataset():
  def __init__(self, data, feature_keys, label_key):
    self.data = data
    self.feature_keys = feature_keys
    self.label_key = label_key

  def __getitem__(self, idx):
    features = [self.data[key][idx] for key in self.feature_keys]
    features = torch.cat(features)
    label = self.data[self.label_key][idx]
    return features, label

  def __len__(self):
    return len(self.data[self.label_key])


class DatasetManager():
  def __init__(self, feature_path, discount=0.9, batch_size=32, shuffle=False):
    indices = range(1, 81)
    self.feature_path = feature_path
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.discount = discount
    self.data = self.load_files(indices)

  def __len__(self):
    return len(self.data)

  def get_dataloader(self, idx, feature_keys, label_key):
      ds =  self.data[idx-1] ## video index range from 1-80
      ds =  DictDataset(ds, feature_keys, label_key)
      return DataLoader(ds, batch_size=self.batch_size, shuffle=self.shuffle)

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
