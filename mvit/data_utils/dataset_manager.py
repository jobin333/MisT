from torchdata.datapipes.iter import FileLister, FileOpener
import torch
import os 
from torch.utils.data import DataLoader
from mvit.data_utils.video_reader import VideoReader

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

    video_path = 'video{:02d}.mp4'.format(self.video_index)
    video_path = os.path.join(self.cholec80_dataset_location, video_path)
    timestamp_path = 'video{:02d}-timestamp.txt'.format(self.video_index)
    timestamp_path = os.path.join(self.cholec80_dataset_location, timestamp_path)

    videoreader = VideoReader(video_path=video_path, timestamp_path=timestamp_path,
                        tubelet_size=self.tubelet_size, 
                        enable_accurate_seek=self.enable_video_reader_accurate_seek,
                        frame_skips=self.frame_skips, debugging=self.debugging, aproximate_keyframe_interval=self.aproximate_keyframe_interval)
    self.current_video_reader = videoreader  ## For debugging purpose
    dataloader = DataLoader(videoreader, batch_size=self.batch_size, shuffle=self.shuffle)
    return dataloader
  

class ModelOutputDatasetManager():
    def __init__(self, file_location='./', train_split=0.8, file_index_start=1, 
                 file_index_end=81,  filename_format='tensors_{}.pt', batch_size=32,
                  lstm_training=False):
        self.file_location = file_location
        self.filename_format = filename_format
        self.file_count = file_index_end - file_index_start
        max_train_index = int(train_split*self.file_count)
        self.train_file_nums = list(range(1, max_train_index))
        self.test_file_nums = list(range(max_train_index, self.file_count+1))
        self.lstm_training = lstm_training 
        self.batch_size = batch_size ## Batch_size for not sequential dataset
        ### If lstm_training enabled, return sequential  unshuffled dataset with size
        ### (sequence_size, batch_size, channels, tublet_size, width, height)
        ### Else return shuffled data with (batch_size, channels, tublet_size, width, height)
        
    def file_number_to_filename(self, file_location, filename_format,  file_num):
        filename =  filename_format.format(file_num)
        datapath = os.path.join(file_location, filename)
        return datapath
    
    def dataset_to_dataloader(self, ds):
        if self.lstm_training:
          dl = torch.utils.data.DataLoader(ds, batch_size=1)
          for x, y in dl:
              yield x.unsqueeze(0), y

        else:
          dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
          for x, y in dl:
              yield x, y
    def filename_to_dataset(self, filename):
        ds = torch.load(filename)
        return self.dataset_to_dataloader(ds)
    
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

