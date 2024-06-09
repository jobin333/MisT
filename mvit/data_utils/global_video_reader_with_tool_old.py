#!/bin/bash

import torchvision
import torch
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import sys


class VideoReader(torch.utils.data.Dataset):
  '''
  Sample Cholec80 video with new sampling factor
  (new_sampling_freq = sampling_freq/sampling_factor) and given size
  using torchvision.io.VideoReader for video extraction. extract_frame function output tensorflow tensor
  containing video and timestamp of the frame.

  ################ Example code

  Expecting preprocessed video dataset with keyframes added to the video at 1 second interval.
  If the key frames are not added in the video the processing time will take much time
  You can use the shell script provided in the example folder to do the same.

  video_reader = VideoReader(video_path=video_path, timestamp_path=timestamp_path,
                              image_shape=image_shape, sampling_factor=sampling_factor)
  cholec80_ds = video_reader.get_dataset()

  '''
  def __init__(self, video_path, timestamp_path, tool_annotations_path=None, tubelet_size=25, 
               frame_skips=0, debugging=False, dataset='cholec80', 
                required_labels = ['phase'] ):
    label_index_col = self.get_label_index_cols(dataset)
    self.surgical_timestamp_df = pd.read_csv(timestamp_path, sep='\t').set_index(label_index_col)
    if 'tool' in required_labels:
      self.surgical_tool_df = pd.read_csv(tool_annotations_path, sep='\t').set_index(label_index_col)
      self.surgical_tool_vocab_list = list(self.surgical_tool_df.columns) 
    # self.surgical_phases = list(self.surgical_timestamp_df.Phase.unique())
    surgical_phases = self.obtain_surg_phases(dataset)
    self.surgical_phase_vocab = build_vocab_from_iterator([surgical_phases])
    self.surgical_phase_vocab_dict = self.surgical_phase_vocab.vocab.get_stoi() 
    self.reader = torchvision.io.VideoReader(video_path, "video")
    self.frame_skips = frame_skips ## Number of frames interleaved while tubelet creation
    self.required_labels = required_labels
    self.video_fps = self.reader.get_metadata()['video']['fps'][0]
    self.video_duration = self.reader.get_metadata()['video']['duration'][0]
    self.tubelet_size = tubelet_size
    self.debugging = debugging ## If debugging enabled set dataloader size to zero
    self.last_pts = 0.0
    self.max_accurate_sequential_search_time = 7.0 ## Sequential search upto 1 second
    self.inter_frame_interval = 0.04 ## 1/fps
    self.seek_offset_time = 0.48 ### for Tool presence

  def obtain_surg_phases(self, dataset):
    cholec80_surg_phases = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting',
                        'GallbladderDissection', 'GallbladderPackaging',
                        'CleaningCoagulation', 'GallbladderRetraction']

    m2cai16_surg_phases = ['TrocarPlacement','Preparation','CalotTriangleDissection'
                           ,'ClippingCutting','GallbladderDissection','GallbladderPackaging'
                           ,'CleaningCoagulation','GallbladderRetraction']
    
    if dataset == 'cholec80':
      surg_phases = cholec80_surg_phases
    elif dataset == 'm2cai16':
      surg_phases = m2cai16_surg_phases
    else:
      raise AttributeError('Only cholec80 and m2cai16 dataset are supported now')
    return surg_phases


  def get_label_index_cols(self, dataset):
    if dataset == 'cholec80':
      index_col = 'Frame'
    elif dataset == 'm2cai16':
      index_col = 'Time'
    else:
      raise AttributeError('Only cholec80 and m2cai16 dataset are supported now')
    return index_col



  def _time_to_timestamp_string(self, t):
    '''
    Convert floating point time to Cholec80 timestamp format.
    '''
    try:
      m, s = divmod(t, 60)
      h, m = divmod(m, 60)
      s = round(s, 2)
      m = round(m)
      h = round(h)
      timestamp_string = '{:02d}:{:02d}:{:05.2f}'.format(h,m,s)
      surgical_phase = self.surgical_timestamp_df.xs(timestamp_string).Phase
    except:
      '''
      For handling a single error situation in m2cai dataset
      '''
      m, s = divmod(t, 60)
      s = round(s, 2)
      m = round(m)
      h = 0
      timestamp_string = '{:02d}:{:02d}:{:05.2f}'.format(h,m,s)
      surgical_phase = self.surgical_timestamp_df.xs(timestamp_string).Phase
    surgical_phase_vocab_index = self.surgical_phase_vocab[surgical_phase]
    return timestamp_string, surgical_phase, surgical_phase_vocab_index

  def _extract_frames_generator(self):
    '''
    Returning frames and timestamp tensorflow tensor.
    '''
    frames = []
    labels = []
    timestamps = []

    for i,f in enumerate(self.reader):
      self.last_pts = f['pts']
      if i%(self.frame_skips+1) != 0:
        continue
      frame = f['data']
      timestamp = f['pts'] + 0.04
      frames.append(frame)
      timestamps.append(timestamp)

      if len(frames) == self.tubelet_size:
        frames = torch.stack(frames)
        frames = frames.permute(1,0,2,3) / 255
        labels = self.retrieve_labels(timestamps)
        return frames, labels


      # tensorflow_frame = tf.image.convert_image_dtype(tensorflow_frame, tf.float32)

  def retrieve_labels(self, timestamps):
    data = []

    if 'phase' in self.required_labels:
      phases = self.get_phase_tensor(timestamps)
      data.append(phases)

    if 'tool' in self.required_labels:
      tools = self.get_tool_tensor(timestamps)
      data.append(tools)

    return data


  def get_phase_tensor(self, timestamps):
    #### Code is not completed
    #### Modification in the code is necessory
    surg_phases = []
    for timestamp in timestamps:
      _, _, surgical_phase_vocab_index = self._time_to_timestamp_string(timestamp)
      surgical_phase_vocab_index = torch.tensor(surgical_phase_vocab_index)
      surg_phases.append(surgical_phase_vocab_index)
    surg_phases = torch.stack(surg_phases)
    return torch.median(surg_phases)
  

  def get_tool_tensor(self, timestamp):
    current_frame_number = int(timestamp*25) ## Tool annotations in 25 frame interval
    tool_annotation_frame_number = current_frame_number - ( current_frame_number % 25 ) ## Tool annotations in 25 frame interval
    tools = self.surgical_tool_df.xs(tool_annotation_frame_number).to_dict()
    tools = list(tools.values())
    tools = torch.tensor(tools)
    return tools 
  # def __iter__(self):
  #   ds = self._extract_frames_generator()
  #   return ds

  def __len__(self):
    '''
    The '__len__' and '__getitem__' method of is used for subclassing torch.utils.data.Dataset.
    This dataset can be shuffled. It is essential fro faster training

    It will return number of items in the dataset.
    '''
    ### If debugging is enabled, it will set the size of the dataloader to 1.
    if self.debugging:
      return 1
    else:
      return int(self.video_duration 
                 - (self.tubelet_size/self.video_fps) - self.seek_offset_time - 1) ## last -1 for safty

  def __getitem__(self, i):
    '''
    Method for subclassing torch.utils.data.Dataset
    The seek function is implemented for only key frames.
    So we are expecting the duration between two key frames is self.aproximate_keyframe_interval = 10 Seconds
    Seek to the time i
    '''
    seek_point = i + self.seek_offset_time
    self.accurate_seek(seek_point)
    return self._extract_frames_generator()



  def accurate_seek(self, seek_location):
    seek_location = seek_location - self.inter_frame_interval  ## Seek to last frame before the required
    seek_location = round(seek_location, 2)
    max_seek = 25*20 ## fps * max_iner_key_frame_duration
    search_offset = seek_location - self.last_pts
    # print('Seek location: {}, search_offset:{}'.format(seek_location, search_offset))
    if search_offset < self.max_accurate_sequential_search_time and search_offset >= 0:
      pts = self.last_pts
      for i in range(max_seek):
        if pts == seek_location:
          return self.reader
        _, pts  = next(self.reader).values()
        self.last_pts = pts

    else:
      self.reader.seek(seek_location, keyframes_only=True)
      for i in range(max_seek):
        _, pts  = next(self.reader).values()
        self.last_pts = pts
        if pts == seek_location:
          return self.reader
    raise Exception('Unable to locate the timestamp')

  def get_vocab(self):
    '''
    Get surgical phase vocabulary
    '''
    return self.surgical_phase_vocab.vocab.get_stoi()    
  
  def show(self, frames):
    '''
    Function to plot frames obtained by videoreader
    '''
    frames = frames.permute(1,0,2,3)[:25]
    img = make_grid(frames, nrow=5).permute(1,2,0)
    plt.imshow(img)
    plt.show()

    
    
if __name__ == '__main__':
  video_path = sys.argv[1]
  timestamp_path = sys.argv[2]
  reader = VideoReader(video_path=video_path, timestamp_path=timestamp_path, dataset='m2cai16')
  frames, labels = reader[600]
  print(labels)
  reader.show(frames)