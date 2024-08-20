import torch
import torchvision

import  matplotlib.pyplot as plt
import pandas as pd



def plot_grid(frames, save_file, show):
  image = torchvision.utils.make_grid(frames, nrow=3)
  image = torchvision.transforms.functional.to_pil_image(image)
  plt.figure(figsize = (40,12))
  fig = plt.imshow(image)
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  plt.savefig(save_file, bbox_inches='tight')
  if show:
    plt.show()


def plot_subplots(frames, labels, save_file, show):
  fig, axes = plt.subplots(2, 3, figsize=(10, 4))  
  axes = axes.flatten()  

  for i, (figure, label) in enumerate(zip(frames, labels)):
      figure = figure.permute((1,2,0)).numpy()
      axes[i].imshow(figure)
      axes[i].set_title(label)
      axes[i].grid(False)  
      axes[i].set_xticks([]) 
      axes[i].set_yticks([]) 


  plt.savefig(save_file, bbox_inches='tight')
  plt.tight_layout()
  if show:
    plt.show()

def generate_video_repr(video_path, save_file, label_path, show=False):
  df = pd.read_csv(label_path, sep='\t')
  reader = torchvision.io.VideoReader(video_path)
  frames = []
  seek_pts = []
  labels = []
  for timestamp in seeks:
    if type(timestamp) == torch.Tensor:
      timestamp = timestamp.item()
    timestamp = int(timestamp)
    reader.seek(timestamp)
    for i in range(500):
      data = next(reader)
      if int(data['pts']*100) == int(timestamp*100):
        break
    frame = data['data']
    # frame = torchvision.transforms.functional.resize(frame, [s for s in frame.shape[1:]])
    pts = data['pts']
    seek_pts.append(pts)
    frames.append(frame)
    label = df.iloc[round(pts*25)].Phase
    labels.append(label)
    # plot_grid(frames, save_file, show)
  plot_subplots(frames, labels, save_file, show)
  return seek_pts







orginal_video_path = 'video01.mp4'
original_filename = 'cholec80-original.pdf'
label_path = 'video01-phase.txt'

# scaled_video_path = 'scaled_video01.mp4'
# scaled_filename = 'cholec80-scaled.jpg'


steps = 6
seeks = torch.linspace(0, 1500, steps=steps)
seeks = [10, 200, 800, 1000, 1520, 1600]




seeks = generate_video_repr(orginal_video_path, original_filename,label_path=label_path, show=True)

print(seeks)
# seeks = generate_video_repr(scaled_video_path, scaled_filename)