import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt 
import torch


def create_phase_color_map():
  red = np.array((1, 0, 0))
  green = np.array((0, 1, 0))
  blue = np.array((0, 0, 1))
  white = red + green + blue

  colors = []
  for i in range(3, 10):
    shade_red = red*0.1*(i) + blue*0.1*(i-3) + green*0.1*(i-3)
    shade_blue = blue*0.1*(i) + red*0.1*(i-3) + green*0.1*(i-3)
    shade_green = green*0.1*(i) + red*0.1*(i-3) + blue*0.1*(i-3)
    shade_cyan= green*0.1*(i) + red*0.1*(i-3) + blue*0.1*(i)

    shade = shade_blue
    colors.append(shade)
  colors.append(white)
  cmap = LinearSegmentedColormap.from_list('phase_map', colors, 8)
  return cmap


def generate_phase_plot(y, linewidth=20, color='white', cmap='copper'):
  plt.imshow(y, aspect='auto', interpolation='none', cmap=cmap)
  cbar = plt.colorbar()
  nums = len(y)
  for i in range(-1, nums):
    plt.axhline(i+0.5, 0, 1000, linewidth=linewidth, color=color)
  plt.xlabel('Time')
  plt.ylabel('Video Number')
  cbar.ax.set_yticklabels(['Preparation', 'CalotTriangleDissection', 'ClippingCutting',
                            'GallbladderDissection', 'GallbladderPackaging',
                            'CleaningCoagulation', 'GallbladderRetraction'],)
  cbar.ax.tick_params(right=False)
  plt.show()


if __name__ == '__main__':
  y = torch.randint(1, 6, (100,100))
  cmap = create_phase_color_map()
  generate_phase_plot(y, cmap=cmap)

