import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt 
import torch
from sklearn.metrics import ConfusionMatrixDisplay


from mvit.metrics.metrics import APRFSJC


class TemporalPhasePlotter():

  def __init__(self, shade_color='blue', colors=None, cmap=None,  dataset_name = 'cholec80', dpi=100):
    '''
    colors : optional,  List of colors; based on this cmap is created, 
            Total classes colors are needed
    cmap: The plot is based on cmap, it is given

    shaded_color: if colors and cmap are not provided, plot will generated based in shade_color
    '''
    
    self.surg_phases = self.get_surgical_phases(dataset_name)
    self.num_classes = len(self.surg_phases)
    if colors is not None:
      colors = colors[:self.num_classes]
      colors = colors + ['white']

    if cmap is None:
      self.cmap = self.create_phase_color_map(shade_color=shade_color, segments=self.num_classes, colors=colors)
    else:
      self.cmap = cmap
    self.dpi = dpi

  def padded_stack(self, data, padding_value):
    padded_data = []
    max_len = max([len(item) for item in data])
    padding = [max_len - len(item) for item in data]
    for pad_len, data in zip(padding, data):
        item = torch.nn.functional.pad(data, (0,pad_len), value=padding_value)
        padded_data.append(item)
    return torch.stack(padded_data)
  
  def get_surgical_phases(self, dataset_name):

    cholec80_surgical_phases = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting',
                            'GallbladderDissection', 'GallbladderPackaging',
                            'CleaningCoagulation', 'GallbladderRetraction']
  
    m2cai16_surgical_phases  = ['TrocarPlacement','Preparation','CalotTriangleDissection'
                              ,'ClippingCutting','GallbladderDissection','GallbladderPackaging'
                              ,'CleaningCoagulation','GallbladderRetraction']
    
    autolaparo_surgical_phases  = ['TrocarPlacement','Preparation','CalotTriangleDissection'
                              ,'ClippingCutting','GallbladderDissection','GallbladderPackaging'
                              ,'CleaningCoagulation','GallbladderRetraction']
    

    if dataset_name == 'cholec80':
      surg_phases = cholec80_surgical_phases
    elif dataset_name =='m2cai16':
      surg_phases = m2cai16_surgical_phases
    elif dataset_name == 'autolaparo':
      surg_phases = autolaparo_surgical_phases
    else:
      raise AttributeError('f{dataset_name} is not supported')
    
    return surg_phases
    

  def create_phase_color_map(self, shade_color='blue', colors=None, segments=7):
    '''
    supported shaded are 'blue', 'green', 'red', 'cyan'
    '''
    red = np.array((1, 0, 0))
    green = np.array((0, 1, 0))
    blue = np.array((0, 0, 1))
    white = red + green + blue

    if colors is None:
      colors = []
      for i in range(10-segments, 10):
        shade = {}
        shade['red'] = red*0.1*(i) + blue*0.1*(i-3) + green*0.1*(i-3)
        shade['blue'] = blue*0.1*(i) + red*0.1*(i-3) + green*0.1*(i-3)
        shade['green'] = green*0.1*(i) + red*0.1*(i-3) + blue*0.1*(i-3)
        shade['cyan'] = green*0.1*(i) + red*0.1*(i-3) + blue*0.1*(i)

        colors.append(shade[shade_color])
        
      colors.append(white)
    cmap = LinearSegmentedColormap.from_list('phase_map', colors, 8)
    return cmap


  def generate_phase_plot(self, path, data_key='slm_metrics_details', 
                          label_key='yt', save_file=None, num_video_files=10,
                          display=True, linewidth=5, seperator_color='white'):
    data = torch.load(path)
    y = data[data_key][label_key][:num_video_files]
    y = self.padded_stack(y, self.num_classes)

    plt.imshow(y, aspect='auto', interpolation='none', cmap=self.cmap)
    cbar = plt.colorbar()
    nums = len(y)
    for i in range(-1, nums):
      plt.axhline(i+0.5, 0, 1000, linewidth=linewidth, color=seperator_color)
    plt.xlabel('Time')
    plt.ylabel('Video Number')
    cbar.ax.set_yticklabels(self.surg_phases)
    cbar.ax.tick_params(right=False)

    if save_file is not None:
      plt.savefig(save_file, bbox_inches='tight', dpi=self.dpi)

    if display:
      plt.show()


def generate_confusion_matrix(config_path, save_path=None, font_size=8, xticks_rotation=90, dpi=100):
    data = torch.load(config_path)
    speculative_confusion = data['slm_metrics_details']['confusion']
    speculative_confusion

    dataset_details = data['dataset_details']
    dataset_name = data['dataset_name']
    surgical_phases = dataset_details[dataset_name]['surgical_phases']
    surgical_phases

    disp = ConfusionMatrixDisplay(speculative_confusion, 
                                  display_labels=surgical_phases )
    plt.rcParams.update({'font.size': font_size})    
    disp.plot(xticks_rotation=xticks_rotation)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    plt.show()



if __name__ == '__main__':

  #

  metric = APRFSJC()
  for i in range(8):
    yt = torch.randint(0, 7, (1000,))
    yp = torch.randint(0, 7, (1000,7))
    metric.update(yp, yt, phase='test')
  metric.compute(phase='test')

  y = metric.metrics['test']['yt']
  y = torch.stack(y, dim=0)
  path = '/workspace/Models3/cholec80_Swin3D_S_0_pt.config'
  colors = ['brown', 'chocolate', 'olive', 'teal', 'steelblue', 'purple', 'mediumvioletred', 'gray', 'navy']
  plotter = TemporalPhasePlotter(colors=colors)
  plotter.generate_phase_plot(path, linewidth=5, label_key='yt')

