import torchvision
import torch
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


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
  def __init__(self, video_path, timestamp_path, tubelet_size=25, 
               frame_skips=0, debugging=False):
    self.surgical_timestamp_df = pd.read_csv(timestamp_path, sep='\t', skiprows=1, names=['Time', 'Phase'])
    self.surgical_timestamp_df.set_index('Time', inplace=True)
    self.surgical_phase_vocab = build_vocab_from_iterator([self.surgical_phases])
    self.surgical_phase_vocab_dict = self.surgical_phase_vocab.vocab.get_stoi() 
    self.reader = torchvision.io.VideoReader(video_path, "video")
    self.frame_skips = frame_skips ## Number of frames interleaved while tubelet creation
    self.video_fps = self.reader.get_metadata()['video']['fps'][0]
    self.video_duration = self.reader.get_metadata()['video']['duration'][0]
    self.tubelet_size = tubelet_size
    self.debugging = debugging ## If debugging enabled set dataloader size to zero
    self.last_pts = 0.0
    self.max_accurate_sequential_search_time = 7.0 ## Sequential search upto 1 second
    self.inter_frame_interval = 0.04 ## 1/fps
    self.seek_offset_time = 0.48 ### for Tool presence



  def _video_pts_to_phase_index(self, t):
    '''
    Convert floating point time to Cholec80 timestamp format.
    '''
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    s = round(s, 2)
    m = round(m)
    h = round(h)
    timestamp_string = '{:02d}:{:02d}:{:05.2f}'.format(h,m,s)
    surgical_phase = self.surgical_timestamp_df.Phase[timestamp_string]
    surgical_phase_vocab_index = self.surgical_phase_vocab[surgical_phase]
    return surgical_phase_vocab_index

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
        labels = self.get_phase_tensor(timestamps)
        return frames, labels


      # tensorflow_frame = tf.image.convert_image_dtype(tensorflow_frame, tf.float32)


  def get_phase_tensor(self, timestamps):
    #### Code is not completed
    #### Modification in the code is necessory
    surg_phases = []
    for timestamp in timestamps:
      surgical_phase_vocab_index = self._video_pts_to_phase_index(timestamp)
      surgical_phase_vocab_index = torch.tensor(surgical_phase_vocab_index)
      surg_phases.append(surgical_phase_vocab_index)
    surg_phases = torch.stack(surg_phases)
    return torch.median(surg_phases)
  

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

    

class Cholec80VideoReader(VideoReader):
  def __init__(self, video_path, timestamp_path, tubelet_size=25, frame_skips=0, debugging=False):
   
    self.surgical_phases  = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting',
                        'GallbladderDissection', 'GallbladderPackaging',
                        'CleaningCoagulation', 'GallbladderRetraction']
    
   
    super().__init__(video_path, timestamp_path, tubelet_size, frame_skips, debugging)



class M2cai16VideoReader(VideoReader):
  def __init__(self, video_path, timestamp_path, tubelet_size=25, frame_skips=0, debugging=False):
   
    self.surgical_phases  = ['TrocarPlacement','Preparation','CalotTriangleDissection'
                           ,'ClippingCutting','GallbladderDissection','GallbladderPackaging'
                           ,'CleaningCoagulation','GallbladderRetraction']
    
   
    super().__init__(video_path, timestamp_path, tubelet_size, frame_skips, debugging)


class AutoLaparoVideoReader(VideoReader):
  def __init__(self, video_path, timestamp_path, tubelet_size=25, frame_skips=0, debugging=False):
   
    self.surgical_phases  = ['TrocarPlacement','Preparation','CalotTriangleDissection'
                           ,'ClippingCutting','GallbladderDissection','GallbladderPackaging'
                           ,'CleaningCoagulation','GallbladderRetraction']
    

    super().__init__(video_path, timestamp_path, tubelet_size, frame_skips, debugging)
    self.surgical_phase_vocab_dict = {phase:idx+1 for idx, phase in enumerate(self.surgical_phases)} 
   

  def _video_pts_to_phase_index(self, t):
    '''
    Convert floating point time to Autolapro timestamp index format.
    '''
    t = round(t)
    surgical_phase_index = self.surgical_timestamp_df.Phase[t]
    return surgical_phase_index

    
if __name__ == '__main__':
  '''
  The following sample codes can be used to test the above classes.
  '''
  pass 

  # ### M2CAI16 Dataset
  # video_path = '/home/jobin/PhD/Datasets/m2cai16/train_dataset/workflow_video_01.mp4'
  # timestamp_path = '/home/jobin/PhD/Datasets/m2cai16/train_dataset/workflow_video_01_timestamp.txt'
  # reader = M2cai16VideoReader(video_path=video_path, timestamp_path=timestamp_path)
  # frames, labels = reader[600]
  # print(labels)
  # reader.show(frames)


  ### Cholec80 Dataset
  # video_path = '/home/jobin/PhD/Datasets/cholec80/videos/video01.mp4'
  # timestamp_path = '/home/jobin/PhD/Datasets/cholec80/videos/video01-timestamp.txt'
  # reader = Cholec80VideoReader(video_path=video_path, timestamp_path=timestamp_path)
  # frames, labels = reader[600]
  # print(labels)
  # reader.show(frames)


  ### AutoLaparo Dataset
  # video_path = '/home/jobin/PhD/Datasets/autolaparo/videos/video01.mp4'
  # timestamp_path = '/home/jobin/PhD/Datasets/autolaparo/labels/label_01.txt'
  # reader = AutoLaparoVideoReader(video_path=video_path, timestamp_path=timestamp_path)
  # frames, labels = reader[60]
  # print(labels)
  # reader.show(frames)




