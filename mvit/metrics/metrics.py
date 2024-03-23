class Accuracy():
  def __init__(self):
    self.datapoints_seen = {'train':0, 'test':0}
    self.correct_prediction = {'train':0, 'test':0}
    self.history_metrics = {'train':[], 'test':[]}
    self.accuracy = 0

  def compute(self, phase):
    accuracy = self.correct_prediction[phase] / self.datapoints_seen[phase]
    self.history_metrics[phase].append(accuracy)
    self.accuracy = accuracy.item()
    self.reset(phase)

  def history(self):
    return self.history_metrics

  def __str__(self):
    return 'Accuracy: {:1.2f}'.format(self.accuracy)

  def update(self, pred, target, phase):
    pred = pred.argmax(-1)
    correct = sum(pred == target)
    self.correct_prediction[phase] += correct
    self.datapoints_seen[phase] += len(pred)

  def reset(self, phase):
    self.datapoints_seen[phase] = 0
    self.correct_prediction[phase] = 0