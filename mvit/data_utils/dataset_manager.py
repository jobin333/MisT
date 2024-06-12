from torchdata.datapipes.iter import FileLister, FileOpener
import torch
import os 
from torch.utils.data import DataLoader
from mvit.data_utils.video_reader import VideoReader, VideoReaderOld
from mvit.logging_utils.logger import logger
import random
import numpy as np

module_logger = logger.getChild(__name__)



class Cholec80DatasetManagerOld():
  '''
  ####### Example
  dm = Cholec80DatasetManager(data_path, tubelet_size, batch_size)
  dataloader = dm.get_dataloader()
  '''

  def __init__(self, cholec80_dataset_location, 
               tubelet_size, batch_size, frame_skips, debugging=False, shuffle=True,
               aproximate_keyframe_interval=10, 
                 enable_video_reader_accurate_seek=False, ## Accurate seek is not recommended, it will slow you down
                 ):
    self.cholec80_dataset_location = cholec80_dataset_location
    self.tubelet_size = tubelet_size
    self.batch_size = batch_size
    self.video_index= 0
    self.dataset_length = 80 #There are 80 vidoes in the Cholec80 dataset
    self.debugging = debugging # If debugging is enabled the dataloader produce only one tubelet
    self.frame_skips = frame_skips # Intra tubelet skips
    self.shuffle = shuffle
    self.enable_video_reader_accurate_seek = enable_video_reader_accurate_seek ## It will slow the system
    self.aproximate_keyframe_interval = aproximate_keyframe_interval

  def __len__(self):
    return self.dataset_length

  def get_dataloader(self, video_index=None):
    '''
    Generate stateful dataloader. Each call will give dataloader based on consicutive video.
    If index is specified, it will give dataloader for that specific indexed video.
    '''
    if video_index is None:
      self.video_index+= 1
      if self.video_index > 80:
        self.video_index= 1
    else:
      self.video_index = video_index

    video_path = 'video{:02d}.mp4'.format(self.video_index)
    video_path = os.path.join(self.cholec80_dataset_location, video_path)
    timestamp_path = 'video{:02d}-timestamp.txt'.format(self.video_index)
    timestamp_path = os.path.join(self.cholec80_dataset_location, timestamp_path)

    videoreader = VideoReaderOld(video_path=video_path, timestamp_path=timestamp_path,
                        tubelet_size=self.tubelet_size, 
                        enable_accurate_seek=self.enable_video_reader_accurate_seek,
                        frame_skips=self.frame_skips, debugging=self.debugging, aproximate_keyframe_interval=self.aproximate_keyframe_interval)
    self.current_video_reader = videoreader  ## For debugging purpose
    dataloader = DataLoader(videoreader, batch_size=self.batch_size, shuffle=self.shuffle)
    return dataloader
  
class Cholec80DatasetManager():
  '''
  ####### Example
  dm = Cholec80DatasetManager(data_path, tubelet_size, batch_size)
  dataloader = dm.get_dataloader()
  '''

  def __init__(self, cholec80_dataset_location, 
               tubelet_size, batch_size, frame_skips, debugging=False, shuffle=True,
               aproximate_keyframe_interval=10, 
                 enable_video_reader_accurate_seek=False, ## Accurate seek is not recommended, it will slow you down
                 ):
    self.cholec80_dataset_location = cholec80_dataset_location
    self.tubelet_size = tubelet_size
    self.batch_size = batch_size
    self.video_index= 0
    self.dataset_length = 80 #There are 80 vidoes in the Cholec80 dataset
    self.debugging = debugging # If debugging is enabled the dataloader produce only one tubelet
    self.frame_skips = frame_skips # Intra tubelet skips
    self.shuffle = shuffle
    self.enable_video_reader_accurate_seek = enable_video_reader_accurate_seek ## It will slow the system
    self.aproximate_keyframe_interval = aproximate_keyframe_interval

  def __len__(self):
    return self.dataset_length

  def get_dataloader(self, video_index=None):
    '''
    Generate stateful dataloader. Each call will give dataloader based on consicutive video.
    If index is specified, it will give dataloader for that specific indexed video.
    '''
    if video_index is None:
      self.video_index+= 1
      if self.video_index > 80:
        self.video_index= 1
    else:
      self.video_index = video_index

    tool_folder = 'tool_annotations'
    video_folder = 'videos'
    timestamp_folder = 'videos'

    video_path = 'video{:02d}.mp4'.format(self.video_index)
    video_path = os.path.join(self.cholec80_dataset_location, video_folder, video_path)
    timestamp_path = 'video{:02d}-timestamp.txt'.format(self.video_index)
    timestamp_path = os.path.join(self.cholec80_dataset_location, timestamp_folder, timestamp_path)
    tool_annotations_path = 'video{:02d}-tool.txt'.format(self.video_index)
    tool_annotations_path = os.path.join(self.cholec80_dataset_location, tool_folder, tool_annotations_path)

    videoreader = VideoReader(video_path=video_path, timestamp_path=timestamp_path, 
                              tool_annotations_path = tool_annotations_path,
                        tubelet_size=self.tubelet_size, 
                        enable_accurate_seek=self.enable_video_reader_accurate_seek,
                        frame_skips=self.frame_skips, 
                        debugging=self.debugging, 
                        aproximate_keyframe_interval=self.aproximate_keyframe_interval)
    self.current_video_reader = videoreader  ## For debugging purpose
    dataloader = DataLoader(videoreader, batch_size=self.batch_size, shuffle=self.shuffle)
    return dataloader
  

class SequentialDataset():
    def __init__(self, dataset, seq_length):
        if len(dataset[0]) == 3:
           self.tool_info = True
        else:
           self.tool_info = False
        self.dataset = dataset
        self.dataset_length = len(dataset) - seq_length - 1 # 1 For safty 
        self.seq_length = seq_length
  
    def __len__(self):
        return self.dataset_length 
  
    def __getitem__(self, idx):
        data_slice = self.dataset[idx:idx+self.seq_length]
        if not self.tool_info:
          x, y = zip(*data_slice)
          x = torch.stack(x)
          y = torch.stack(y)
          y = y[self.seq_length // 2]
          return x, y
        else:
          x, y, z = zip(*data_slice)
          x = torch.stack(x)
          y = torch.stack(y)
          z = torch.stack(z)
          y = y[self.seq_length // 2]
          z = z[self.seq_length // 2]
          return x, y, z           


class ModelOutputDatasetManager():
    '''
    Make batch_first to True for using transformer model written with this package
    '''
    def __init__(self, file_location, train_split=0.9, file_index_start=1, single_batch=False,
                 file_index_end=81,  filename_format='tensors_{}.pt', batch_size=32, device=None,
                  lstm_training=False, mapping_fn = None, shuffle=True, seq_length=None, batch_first=True,
                  tool_info=True, tool_type=torch.float, tool_unity_norm=False):
        if mapping_fn is None:
          self.mapping_fn = lambda x: x
        else:
          self.mapping_fn = mapping_fn
        
        self.tool_info = tool_info
        self.tool_unity_norm = tool_unity_norm
        self.tool_type = tool_type
        self.file_location = file_location
        self.filename_format = filename_format
        self.file_count = file_index_end - file_index_start
        max_train_index = int(train_split*self.file_count)
        ## training_file_number and test_file_number are not using
        ## It will removed in future
        self.train_file_nums = list(range(1, max_train_index))
        self.test_file_nums = list(range(max_train_index, self.file_count+1))
        self.lstm_training = lstm_training 
        self.shuffle = shuffle
        self.device = device
        self.batch_first = batch_first
        self.single_batch = single_batch ##### Obtain entair dataset once. No batches
        self.batch_size = batch_size ## Batch_size for not sequential dataset
        ### If lstm_training enabled, return sequential  unshuffled dataset with size
        ### (sequence_size, batch_size, channels, tublet_size, width, height)
        ### Else return shuffled data with (batch_size, channels, tublet_size, width, height)
        self.seq_length = seq_length ## If it is not None, get_dataloader output sequential dataset for transformer training
    def file_number_to_filename(self, file_location, filename_format,  file_num):
        filename =  filename_format.format(file_num)
        datapath = os.path.join(file_location, filename)
        return datapath
    
    def dataset_to_dataloader(self, ds):
        if self.lstm_training:
          dl = torch.utils.data.DataLoader(ds, batch_size=1)
          for x, y in dl:
              x = x.detach()
              y = y.detach()
              x = self.mapping_fn(x)
              yield x.unsqueeze(0), y

        else:
          batch_size=self.batch_size
          if self.single_batch:
             batch_size = len(ds)
          dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=self.shuffle)
          for x, y in dl:
              x = x.detach()
              y = y.detach()
              x = self.mapping_fn(x)
              yield x, y


    def dataset_to_dataloader_sequential(self, ds):
        seq_ds = SequentialDataset(ds, self.seq_length)
        dl = DataLoader(seq_ds, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=False)
        # if not self.batch_first:
        #   for x,y in dl:
        #       x = x.detach()
        #       y = y.detach()
        #       x = x.permute(1,0,2)
        #       yield x, y

        # if  self.batch_first:
        #   for x,y in dl:
        #       x = x.detach()
        #       y = y.detach()
        #       yield x, y
        if self.tool_info:
          for frame,phase,tool in dl:
            frame = frame.detach()
            phase = phase.detach()
            tool = tool.detach().to(self.tool_type)
            if not self.batch_first:
              frame = frame.permute(1,0,2)
            if self.tool_unity_norm:
              tool = tool * 2 - 1
            yield frame, phase, tool 

        if not self.tool_info:
          for frame,phase in dl:
            frame = frame.detach()
            phase = phase.detach()
            if not self.batch_first:
              frame = frame.permute(1,0,2)
            yield frame, phase

    def filename_to_dataset(self, filename):
        module_logger.debug('Generating dataset for file {}'.format(filename))
        ds = torch.load(filename,  map_location=self.device)
        if self.seq_length is None:
          return self.dataset_to_dataloader(ds)
        else:
           return self.dataset_to_dataloader_sequential(ds)
    
    def get_dataloader(self, file_num):
        filename = self.file_number_to_filename(self.file_location, self.filename_format, file_num)
        return self.filename_to_dataset(filename)
    
    def get_train_dataloader(self):
        file_num = self.train_file_nums.pop()
        self.train_file_nums.insert(0,file_num)
        filename = self.file_number_to_filename(self.file_location, self.filename_format, file_num)
        dataloader = self.filename_to_dataloader(filename)
        return dataloader
        
    def get_test_dataloader(self):
        file_num = self.test_file_nums.pop()
        self.test_file_nums.insert(0,file_num)
        filename = self.file_number_to_filename(self.file_location, self.filename_format, file_num)
        dataloader = self.filename_to_dataloader(filename)
        return dataloader
        
    def __len__(self):
        return self.file_count


    def __getitem__(self, index):
       return self.get_dataloader(index+1)
    
    def get_hmm_iterator(self, training=True):
      if training:
        file_nums = self.train_file_nums
      else:
        file_nums = self.test_file_nums

      self.single_batch = True ## Making sure that entair data is produced continuesly
      for i in file_nums:
          ds  =  self.get_dataloader(i)
          for x,y in ds:
             break
          x = x.reshape(1, -1, 7)
          yield x, y



class ConcatFeatureDatasetManager():
  def __init__(self, feature_path, discount=0.9):
    indices = range(1, 81)
    train_indices = np.random.choice(indices, 70, replace=False)
    test_indices = np.setxor1d(indices, train_indices)
    # return np.sort(train_indices), np.sort(test_indices)
    self.discount = discount
    self.train_data = self.load_files(train_indices)
    self.train_data = self.concat_file_data(self.train_data)
    self.test_data = self.load_files(test_indices)
    self.test_data = self.concat_file_data(self.test_data)
    self.feature_path = feature_path

  def get_dataset(self, keys, train=True):
    self.keys = keys
    self.data = (self.train_data if train else self.test_data)
    return self

  def __len__(self):
    return self.data['feature'].shape[0]

  def __getitem__(self, idx):
      return tuple(self.data[key][idx] for key in self.keys)

  def load_files(self, indices):
    file_paths = [os.path.join(self.feature_path, f'tensors_{i}.pt') for i in indices]
    files = [torch.load(path) for path in file_paths ]
    data = [self.add_cumsum_info(file) for file in files]
    return data


class SimpleModelOuptutDatasetManager():
    def __init__(self,features_save_loc, seq_length=30, seq_delay=None, enable_sequence=True):
        self.enable_sequence = enable_sequence
        self.seq_length = seq_length
        if seq_delay is None:
            self.seq_delay = self.seq_length // 2
        else:
            self.seq_delay = seq_delay
        self.features_save_loc = features_save_loc
        self.data_list = [self.get_dataset(idx)for idx in range(1,81)]

    def generate_stack(self, x):
        stack_list = [x]
        for i in range(self.seq_length-1):
            shifted_x = torch.roll(stack_list[-1], 1, dims=0)
            shifted_x[0] = 0.
            stack_list.append(shifted_x)
        stack = torch.stack(stack_list).permute(1,0,2)
        return stack

    def get_dataset(self, idx):
        path = f'tensors_{idx}.pt'
        path = os.path.join(self.features_save_loc, path)
        ds = torch.load(path, map_location=torch.device('cpu'))
        x, y = zip(*ds)
        x = torch.stack(x)
        y = torch.stack(y)
        return x,y 

    def get_dataloader(self, idx):
        x, y = self.data_list[idx-1]
        if not self.enable_sequence:
           return [(x, y)]
        data_length = len(y)
        x = self.generate_stack(x)
        x = x[self.seq_delay:]
        y = y[:data_length - self.seq_delay]
        return [(x, y)]

  
