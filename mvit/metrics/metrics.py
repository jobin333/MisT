import torch 

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import jaccard_score, accuracy_score

class Accuracy():
  def __init__(self):
    self.datapoints_seen = {'train':0, 'test':0}
    self.correct_prediction = {'train':0, 'test':0}
    self.history_metrics = {'train':[], 'test':[]}
    self.accuracy = 0.0
    self.best_eval_accuracy = 0.0
    self.name = 'accuracy'

  def value(self):
    return self.best_eval_accuracy

  def compute(self, phase):
    accuracy = self.correct_prediction[phase] / self.datapoints_seen[phase]
    self.history_metrics[phase].append(accuracy)
    self.accuracy = accuracy.item()
    if phase == 'test':
      self.best_eval_accuracy = self.accuracy
    self.reset(phase)

  def history(self):
    return self.history_metrics

  def __str__(self):
    return 'Accuracy: {:1.3f}'.format(self.accuracy)

  def update(self, pred, target, phase):
    pred = pred.argmax(-1)
    correct = sum(pred == target)
    self.correct_prediction[phase] += correct
    self.datapoints_seen[phase] += len(pred)

  def reset(self, phase):
    self.datapoints_seen[phase] = 0
    self.correct_prediction[phase] = 0


class ConfusionMatrix():
  def __init__(self):
    self.history = {'train':{yt:torch.tensor([]), yp:torch.tensor([])}, 
                    'test':{yt:torch.tensor([]), yp:torch.tensor([])}}
    self.metrics = {'train': None, 'test':None}
    self.last_phase = None

  def update(self, pred, target, phase):
    yt = target
    yp = pred.argmax(-1)
    self.history[phase]['yt'] = torch.cat( (self.history[phase]['yt'], yt) )
    self.history[phase]['yp'] = torch.cat( (self.history[phase]['yp'], yp) )

  def compute(self, phase):
    self.last_phase = phase
    yt = self.history[phase]['yt']
    yp = self.history[phase]['yp']
    self.metrics[phase] = confusion_matrix(yt, yp)
    self.reset()
    return self.metrics[phase]
  
  def __str__(self):
    data = self.metrics[self.last_phase]
    return str(data)
  
  def reset(self, phase):
    self.history[phase]['yt'] = torch.tensor([])
    self.history[phase]['yp'] = torch.tensor([])



class APRFSJC():
  '''
  Generate 'accuracy', 'precision', 'recall', 'fscore', 'support', 'jaccard', 'confusion'
  It needs sklearn library
  Object.value() will return the accuracy of last test phase
  '''
  def __init__(self):
    metrics_names = ['accuracy', 'precision', 'recall', 'fscore', 'support',
                      'jaccard', 'confusion', 'yt', 'yp']
    train_dict = {name : None for name in metrics_names}
    test_dict = {name : None for name in metrics_names}
  
    self.history = {'train':{'yt':[], 'yp':[]}, 
                    'test':{'yt':[], 'yp':[]}}
    
    self.metrics = {'train':train_dict, 'test': test_dict }
    self.last_phase = None


  def update(self, pred, target, phase):
    yt = target
    yp = pred.argmax(-1)

    yt = yt.cpu().detach()
    yp = yp.cpu().detach()
    
    self.history[phase]['yt'].append(yt)
    self.history[phase]['yp'].append(yp)

  def compute(self, phase):
    self.last_phase = phase
    if phase == 'test':
      test_yt = self.history[phase]['yt']
      test_yp = self.history[phase]['yp']

    yt = torch.cat(self.history[phase]['yt'])
    yp = torch.cat(self.history[phase]['yp'])

    confusion = confusion_matrix(yt, yp)
    precision, recall, fscore, support = precision_recall_fscore_support(yt, yp)
    jaccard = jaccard_score(yt, yp, average=None)
    accuracy = accuracy_score(yt, yp)

    self.metrics[phase]['precision'] = precision
    self.metrics[phase]['recall'] = recall
    self.metrics[phase]['fscore'] = fscore
    self.metrics[phase]['support'] = support
    self.metrics[phase]['jaccard'] = jaccard
    self.metrics[phase]['confusion'] = confusion
    self.metrics[phase]['accuracy'] = accuracy
    self.metrics[phase]['yt'] = test_yt
    self.metrics[phase]['yp'] = test_yp

    self.reset(phase)
    return accuracy
    
  
  def __str__(self):
    data = self.metrics[self.last_phase]
    return str(data)
  
  def reset(self, phase):
    self.history[phase]['yt'] = []
    self.history[phase]['yp'] = []


  def value(self):
    return self.metrics['test']['accuracy']



if __name__ == '__main__':
  metric = APRFSJC()
  for i in range(100):
    yt = torch.randint(0, 6, (10,))
    yp = torch.randn((10, 7))  
    metric.update(yp, yt, phase='test')
  metric.compute(phase='test')
  print(metric)
  print(metric.value())

