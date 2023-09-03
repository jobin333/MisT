import io
import ipywidgets
import imageio
from IPython.display import display 

class TubletDatasetViewer():
  '''
  Receives tublet images and labels.
  Produce gif animated images

  ######### Sample Program
  cholec80_dm = Cholec80DatasetManager()
  ds = cholec80_dm.get_tubelet_dataset('test3.tfrecord')
  viewer = TubletDatasetViewer()
  viewer.view(ds)
  plt.show()
  '''
  def __init__(self, samples_count=4):
    self.samples_count = samples_count

  def _make_box_for_grid(self, image_widget, fit):
    # Make the caption
    if fit is not None:
        fit_str = "'{}'".format(fit)
    else:
        fit_str = str(fit)

    h = ipywidgets.HTML(value="" + str(fit_str) + "")

    # Make the green box with the image widget inside it
    boxb = ipywidgets.widgets.Box()
    boxb.children = [image_widget]

    # Compose into a vertical box
    vb = ipywidgets.widgets.VBox()
    vb.layout.align_items = "center"
    vb.children = [h, boxb]
    return vb

  def view(self, dataloader, fps=1):
    '''
    A function to view the tubelet dataset
    '''
    tubelets, labels = next(iter(dataloader))
    boxes = []
    for tubelet, label in zip(tubelets, labels):
      tubelet = tubelet.permute(0,2,3,1)
      # if i>=self.samples_count:
      #   break
      with io.BytesIO() as gif:
        imageio.mimsave(gif, tubelet.numpy(), "GIF")
        gif_value = gif.getvalue()

      ib = ipywidgets.widgets.Image(value=gif_value, width=100, height=100)
      boxes.append(self._make_box_for_grid(ib, label))

    w = ipywidgets.widgets.GridBox(
        boxes, layout=ipywidgets.widgets.Layout(grid_template_columns="repeat(5, 200px)")
    )
    display(w)

