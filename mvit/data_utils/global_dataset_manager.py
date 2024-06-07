import os
from torch.utils.data import DataLoader

# from mvit.data_utils.global_video_reader import VideoReader
from mvit.data_utils.global_video_reader import Cholec80VideoReader, M2cai16VideoReader, AutoLaparoVideoReader


class VideoDatasetManager():
  '''
  ####### Example
  dm = Cholec80DatasetManager(data_path, tubelet_size, batch_size)
  dataloader = dm.get_dataloader()
  '''

  def __init__(self, dataset_location,batch_size,  shuffle=True, tubelet_size=25, 
               frame_skips=0, debugging=False, dataset='cholec80', 
                required_labels = ['phase']  ):

    self.dataset = dataset
    self.required_labels = required_labels
    self.dataset_location = dataset_location
    self.tubelet_size = tubelet_size
    self.batch_size = batch_size
    self.dataset_length = 80 #There are 80 vidoes in the Cholec80 dataset
    self.debugging = debugging # If debugging is enabled the dataloader produce only one tubelet
    self.frame_skips = frame_skips # Intra tubelet skips
    self.shuffle = shuffle

  def __len__(self):
    return self.dataset_length
  
  def get_cholec80_paths(self, video_index):
    tool_folder = 'tool_annotations'
    video_folder = 'videos'
    timestamp_folder = 'videos'

    video_path = 'video{:02d}.mp4'.format(video_index)
    video_path = os.path.join(self.dataset_location, video_folder, video_path)

    timestamp_path = 'video{:02d}-timestamp.txt'.format(video_index)
    timestamp_path = os.path.join(self.dataset_location, timestamp_folder, timestamp_path)

    if 'tool' in self.required_labels:
      tool_annotations_path = 'video{:02d}-tool.txt'.format(video_index)
      tool_annotations_path = os.path.join(self.dataset_location, tool_folder, tool_annotations_path)
    else:
      tool_annotations_path = None

    return video_path, timestamp_path, tool_annotations_path
  

  def get_autolaparo_paths(self, video_index):
    tool_folder = 'tool_annotations'
    video_folder = 'videos'
    timestamp_folder = 'labels'

    video_path = '{:02d}.mp4'.format(video_index)
    video_path = os.path.join(self.dataset_location, video_folder, video_path)

    timestamp_path = 'label_{:02d}.txt'.format(video_index)
    timestamp_path = os.path.join(self.dataset_location, timestamp_folder, timestamp_path)

    if 'tool' in self.required_labels:
      tool_annotations_path = 'video{:02d}-tool.txt'.format(video_index)
      tool_annotations_path = os.path.join(self.dataset_location, tool_folder, tool_annotations_path)
    else:
      tool_annotations_path = None

    return video_path, timestamp_path, tool_annotations_path
  
  
  def get_m2cai16_paths(self, video_index, training_phase=True):
    tool_annotations_path = None
    if training_phase:
      folder_name = 'train_dataset'
      video_path = 'workflow_video_{:02d}.mp4'.format(video_index)
      timestamp_path = 'workflow_video_{:02d}_timestamp.txt'.format(video_index)
    else:
      folder_name = 'train_dataset'
      video_path = 'test_workflow_video_{:02d}.mp4'.format(video_index)
      timestamp_path = 'test_workflow_video_{:02d}_timestamp.txt'.format(video_index)

    video_path = os.path.join(self.dataset_location, folder_name, video_path)
    timestamp_path = os.path.join(self.dataset_location, folder_name, timestamp_path)

    if 'tool' in self.required_labels:
      tool_annotations_path = None  ## Not implemented

    return video_path, timestamp_path, tool_annotations_path

  def get_dataloader(self, video_index, training_phase=None):
    '''
    Generate stateful dataloader. Each call will give dataloader based on consicutive video.
    If index is specified, it will give dataloader for that specific indexed video.
    '''

    if self.dataset == 'cholec80':
      paths = self.get_cholec80_paths(video_index)
      video_path, timestamp_path, tool_annotations_path = paths
      videoreader = Cholec80VideoReader(video_path, timestamp_path,
                               tubelet_size=25, frame_skips=0, debugging=False)

    
    elif self.dataset == 'm2cai16':
      paths = self.get_m2cai16_paths(video_index, training_phase)
      video_path, timestamp_path, tool_annotations_path = paths
      videoreader = M2cai16VideoReader(video_path, timestamp_path,
                               tubelet_size=25, frame_skips=0, debugging=False)

    elif self.dataset == 'autolaparo':
      paths = self.get_autolaparo_paths(video_index)
      video_path, timestamp_path, tool_annotations_path = paths
      videoreader = AutoLaparoVideoReader(video_path, timestamp_path,
                               tubelet_size=25, frame_skips=0, debugging=False)
    
    else:
      raise AttributeError('Only cholec80, m2cai16, autolaparo are supported')


    self.current_video_reader = videoreader  ## For debugging purpose
    dataloader = DataLoader(videoreader, batch_size=self.batch_size, shuffle=self.shuffle)
    return dataloader
  

if __name__ == '__main__':
  pass
  # dataset_location = '/home/jobin/PhD/Datasets/Scaled/Cholec80'
  # dataset_name = 'cholec80'

  # dataset_location = '/home/jobin/PhD/Datasets/Scaled/M2CAI16'
  # dataset_name = 'm2cai16'

  # dataset_location = '/home/jobin/PhD/Datasets/Scaled/Autolaparo'
  # dataset_name = 'autolaparo'

  # dm = VideoDatasetManager(dataset_location, batch_size=4,  shuffle=False,
  #               tubelet_size=25, frame_skips=0, debugging=False, dataset=dataset_name)
  
  # dl = dm.get_dataloader(1, training_phase=True)
  # for (x,y) in dl:
  #   print(y)