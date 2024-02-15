import torch
import numpy as np
import collections 
from sklearn import metrics
from matplotlib import pyplot as plt

def generate_sample_yt_yp():
  p1 = torch.randn((1024, 7))
  p2 = torch.randn((1024, 7))

  yt = p1.argmax(-1)
  yp = (p1+p2*.6).argmax(-1)
  p = np.random.randint(0, 7, size=10)
  p = collections.Counter(p)

  y = [0]*p[0]*100 + [1]*p[1]*100 + [2]*p[2]*100 + [3]*p[3]*100 + [4]*p[4]*100 + [5]*p[5]*100 + [6]*p[6]*100 + [8]*100

  return y, y


def generate_analysis_report(yt, yp, save_path=None, print_report=True):

  ## Code for calculating Precision, Recall, F1 Score, Support, Jaccard
  score = {}
  score['precision'], score['recall'], score['fscore'], score['support'] = metrics.precision_recall_fscore_support(yt, yp)
  score['jaccard'] = metrics.jaccard_score(yt, yp, average=None)

  ## Printing basic reports
  report = metrics.classification_report(yt, yp, digits=3)
  if print_report:
    print(report)

  df = pd.DataFrame.from_dict(score)

  if save_path is not None:
    df.to_csv(save_path)

  return df


# Confusion Matrix

def generate_confusion_matrix(yt, yp, save_path=None, show=True, cmap='Blues'):
  '''
    Function to show and save the confusion matrix to the file given in the path
    Use proper extension in the path for the required image format
  '''
  confusion_matrix = metrics.confusion_matrix(yt, yp)
  plt.imshow(confusion_matrix, cmap=cmap)
  plt.colorbar()
  if save_path is not None:
    plt.savefig(save_path)
  if show:
    plt.show()
  else:
    plt.close()

yt, yp = generate_sample_yt_yp()
generate_analysis_report(yt, yp, print_report=False, save_path='test.csv')
generate_confusion_matrix(yt, yp, save_path='cmatrix.jpg', show=False, cmap='Blues')
