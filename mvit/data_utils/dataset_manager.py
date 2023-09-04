from torchdata.datapipes.iter import FileLister, FileOpener
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
               tubelet_size, batch_size, frame_skips, debugging=False):
    self.cholec80_dataset_location = cholec80_dataset_location
    self.tubelet_size = tubelet_size
    self.batch_size = batch_size
    self.video_index= 0
    self.dataset_length = 80 #There are 80 vidoes in the Cholec80 dataset
    self.debugging = debugging # If debugging is enabled the dataloader produce only one tubelet
    self.frame_skips = frame_skips # Intra tubelet skips

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
                        frame_skips=self.frame_skips, debugging=self.debugging)
    dataloader = DataLoader(videoreader, batch_size=self.batch_size, shuffle=True)
    return dataloader