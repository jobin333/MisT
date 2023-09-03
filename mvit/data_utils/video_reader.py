import torchvision
import torch
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator


class VideoReader(torch.utils.data.Dataset):
  '''
  Sample Cholec80 video with new sampling factor
  (new_sampling_freq = sampling_freq/sampling_factor) and given size
  using torchvision.io.VideoReader for video extraction. extract_frame function output tensorflow tensor
  containing video and timestamp of the frame.

  ################ Example code
  video_reader = VideoReader(video_path=video_path, timestamp_path=timestamp_path,
                              image_shape=image_shape, sampling_factor=sampling_factor)
  cholec80_ds = video_reader.get_dataset()
  '''
  def __init__(self, video_path, timestamp_path, sampling_factor, tubelet_size, aproximate_keyframe_interval = 10):
    self.surgical_timestamp_df = pd.read_csv(timestamp_path, sep='\t').set_index('Frame')
    # self.surgical_phases = list(self.surgical_timestamp_df.Phase.unique())
    self.surgical_phases = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting',
                            'GallbladderDissection', 'GallbladderPackaging',
                            'CleaningCoagulation', 'GallbladderRetraction']
    self.surgical_phase_vocab = build_vocab_from_iterator([self.surgical_phases])
    self.reader = torchvision.io.VideoReader(video_path, "video")
    self.sampling_factor = sampling_factor

    self.video_fps = self.reader.get_metadata()['video']['fps'][0]
    self.video_duration = self.reader.get_metadata()['video']['duration'][0]
    self.tubelet_size = tubelet_size
    self.aproximate_keyframe_interval = aproximate_keyframe_interval
  def _time_to_timestamp_string(self, t):
    '''
    Convert floating point time to Cholec80 timestamp format.
    '''
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    s = round(s, 2)
    m = round(m)
    h = round(h)
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
    for i,f in enumerate(self.reader):
      if i%self.sampling_factor != 0:
        continue
      frame = f['data']
      timestamp = f['pts'] + 0.04
      # tensorflow_frame = tf.image.convert_image_dtype(tensorflow_frame, tf.float32)
      _, _, surgical_phase_vocab_index = self._time_to_timestamp_string(timestamp)
      surgical_phase_vocab_index = torch.tensor(surgical_phase_vocab_index)
      frames.append(frame)
      labels.append(surgical_phase_vocab_index)
      if len(frames) == self.tubelet_size:
        frame_tensor = torch.stack(frames)
        frame_tensor = frame_tensor.permute(1,0,2,3) / 255
        label_tensor = torch.stack(labels)
        frames = []
        labels = []
        return frame_tensor, torch.median(label_tensor)


  # def __iter__(self):
  #   ds = self._extract_frames_generator()
  #   return ds

  def __len__(self):
    '''
    The '__len__' and '__getitem__' method of is used for subclassing torch.utils.data.Dataset.
    This dataset can be shuffled. It is essential fro faster training

    It will return number of items in the dataset.
    '''
    # return 1 ################################################ For debugging
    return int(self.video_duration / self.aproximate_keyframe_interval)

  def __getitem__(self, i):
    '''
    Method for subclassing torch.utils.data.Dataset
    The seek function is implemented for only key frames.
    So we are expecting the duration between two key frames is self.aproximate_keyframe_interval = 10 Seconds
    '''
    self.reader.seek(i*self.aproximate_keyframe_interval, keyframes_only=True)
    return self._extract_frames_generator()
