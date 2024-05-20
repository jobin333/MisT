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
    return 'Accuracy: {:1.3f}'.format(self.accuracy)

  def update(self, pred, target, phase):
    pred = pred.argmax(-1)
    correct = sum(pred == target)
    self.correct_prediction[phase] += correct
    self.datapoints_seen[phase] += len(pred)

  def reset(self, phase):
    self.datapoints_seen[phase] = 0
    self.correct_prediction[phase] = 0





### On Development - printing class wise accuracy
class Accuracy2():
  def __init__(self):
    self.datapoints_seen = {'train':0, 'test':0}
    self.correct_prediction = {'train':0, 'test':0}
    self.history_metrics = {'train':[], 'test':[]}
    self.correct_prediction_list = {'train':[], 'test':[]}
    self.datapoints_seen_list = {'train':[], 'test':[]}
    self.accuracy = 0

  def _gen_class_correct_prediction(predicted, correct, class_idx):
    class_items = correct == class_idx
    predicted_items = predicted == class_idx
    class_count = sum(class_items)
    correct_count = sum(torch.logical_and(class_items, predicted_items))
    return class_count, correct_count

  def compute(self, phase):
    accuracy = self.correct_prediction[phase] / self.datapoints_seen[phase]
    self.history_metrics[phase].append(accuracy)
    self.accuracy = accuracy.item()
    self.reset(phase)

  def history(self):
    return self.history_metrics

  def __str__(self):
    str1 = 'Overall Accuracy: {:1.2f}'.format(self.accuracy)
    str2 = 'Phase Accuracy: {:1.2f}'.format(self.accuracy)
    return f'{str1}\n{str2}'

  def update(self, pred, target, phase):
    pred = pred.argmax(-1)
    correct = sum(pred == target)
    self.correct_prediction[phase] += correct
    self.datapoints_seen[phase] += len(pred)

  def reset(self, phase):
    self.datapoints_seen[phase] = 0
    self.correct_prediction[phase] = 0
    self.datapoints_seen_list[phase] = []
    self.correct_prediction_list[phase] = []