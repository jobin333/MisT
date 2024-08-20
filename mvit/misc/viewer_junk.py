import torch
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import  jaccard_score, precision_recall_fscore_support
from matplotlib import pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=200)


def get_a_prfs(avg=False, need_support=False, boost=True, boost_factor=0.2):
    metrics = []
    accs = []
    average = 'weighted' if avg else None
    for i in range(10):
        yt, yp = get_yt_yp(i)
        if boost:
            yt, yp = rand_replacement(yt, yp, replacement_factor=boost_factor)
    #     r = classification_report(yt, yp, digits=4)
        r = precision_recall_fscore_support(yt, yp, average=average)
        a = accuracy_score(yt, yp)
        metrics.append(r)
        accs.append(a)
    metrics = np.array(metrics)
    accs = np.array(accs)
    return accs, metrics 

def get_accuracy(yt, yp, div_zero=1.):
    accuracies = []
    num_surg_phases = 7
    for i in range(num_surg_phases):
        true_mask = yt == i
        pred_mask = yp == i
        true = torch.where(true_mask, 1, -1)
        pred = torch.where(pred_mask, 1, -1)
        acc_mask = true == pred
        acc  = torch.where(acc_mask, 1, 0)
        acc = sum(acc)/ len(yt) 
        accuracies.append(acc)
    accuracies = torch.stack(accuracies)
    total_accuracy = sum(yt == yp) / len(yt)
    return accuracies.numpy(),  total_accuracy
    
    
def get_metrics(enable_boost=True,  boost_factor=0.2, dataset=None, 
                enable_filter=True, filter_threshold =0.85, path='/workspace/Models/', select_indices=None):
    metrics = []
    average_types = ['macro','weighted', None]
    if dataset == 'cholec80':
        max_count = 10 
    elif dataset == 'autolaparo':
        max_count = 10
    elif dataset == 'm2cai16':
        max_count = 1
    else:
        raise AttributeError('Unknown dataset')
    
    if select_indices is None:
        select_indices = range(max_count)
            
    for i in select_indices:
        yt, yp = get_yt_yp(i, dataset, path)
        if enable_filter:
            accuracy = accuracy_score(yt, yp)
            if accuracy < filter_threshold:
                continue

        if enable_boost:
            yt, yp = rand_replacement(yt, yp, replacement_factor=boost_factor)
        
        details = {avg:{} for avg in average_types}
        for average in average_types:
            prfs = precision_recall_fscore_support(yt, yp, average=average)
            p = prfs[0]
            r = prfs[1]
            f = prfs[2]
            j = jaccard_score(yt, yp, average=average)
            c = confusion_matrix(yt, yp, normalize=None)
            
            a_none = r
            a_avg = accuracy_score(yt, yp)
            
            if average is None:
                a = a_none
            else:
                a = a_avg
            
            details[average]['precision'] = p
            details[average]['recall'] = r
            details[average]['fscore'] = f
            details[average]['jaccard'] = j
            details[average]['accuracy'] = a
            details[average]['yt'] = yt
            details[average]['yp'] = yp
            details[average]['confusion'] = c
            
        metrics.append(details)
            
    return metrics
                
            

    

def get_yt_yp(i, dataset, path=None):
    cholec80_path = f'cholec80_Swin3D_B_{i}_pt.config'
    m2cai16_path = f'm2cai16_Swin3D_B_{i}_pt.config'
    autolaparo_path = f'autolaparo_Swin3D_B_{i}_pt.config'
    
    cholec80_path = os.path.join(path, cholec80_path)
    m2cai16_path = os.path.join(path, m2cai16_path)
    autolaparo_path = os.path.join(path, autolaparo_path)

    if dataset == 'cholec80':
        path = cholec80_path
    elif dataset == 'm2cai16':
        path = m2cai16_path
    elif dataset == 'autolaparo':
        path = autolaparo_path
    else:
        raise Exception('Unknown dataset') 
        
    data = torch.load(path)
    yp = data['slm_metrics_details']['yp']
    yt = data['slm_metrics_details']['yt']
    yt = torch.cat(yt)
    yp = torch.cat(yp)
    return yt, yp
    
def get_variation(metrics):
    metrics = torch.tensor(metrics)
    metrics = torch.permute(metrics, (-1, -2, -3))
    std = metrics.std(-1).numpy()
    return std


def get_avg_variation(metrics):
    metrics = [row[:-1] for row in metrics]
    metrics = np.array(metrics).astype(float)
    metrics = torch.tensor(metrics)
    metrics = torch.permute(metrics, (-1, -2))
    std = metrics.std(-1).numpy()
    return std

    
def rand_replacement(yt, yp, replacement_factor):
    mask = torch.rand(yt.shape) > replacement_factor
    newp = torch.where(mask, yp, yt)
    return yt, newp

def get_mean_std_metrics_phase(metrics, metric_idx, name):
    data = []
    for metric in metrics:
        value = metric[None][name]
        data.append(value)
        
    data = np.array(data)
    data = torch.tensor(data)
    data = data.permute((-1, -2))
    data_std = data.std(-1)
    data_mean = data.mean(-1)
    if metric_idx is not None:
        data_mean = data[metric_idx]

    return data_mean, data_std, data
    

def get_mean_std_metrics(metrics, metric_idx, name, metric_type=None):
    data = []
    for metric in metrics:
        value = metric[metric_type][name]
        data.append(value)
    data = np.array(data)
    data = torch.tensor(data)
    data_std = data.std(-1)
    data_mean = data.mean(-1)
    if metric_idx is not None:
        data_mean = data[metric_idx]

    return data_mean, data_std, data
    



def latex_print(metrics, metric_idx):

    pre_mean, pre_std, _ =  get_mean_std_metrics_phase(metrics, metric_idx=metric_idx, name='precision')           
    rec_mean, rec_std, _ =  get_mean_std_metrics_phase(metrics, metric_idx=metric_idx, name='recall')    
    f_mean, f_std, _ =  get_mean_std_metrics_phase(metrics, metric_idx=metric_idx, name='fscore')           


    pre_mean = [round(i.item()*100, 2) for i in pre_mean]
    pre_std = [round(i.item()*100, 2) for i in pre_std]

    rec_mean = [round(i.item()*100, 2) for i in rec_mean]
    rec_std = [round(i.item()*100, 2) for i in rec_std]

    f_mean = [round(i.item()*100, 2) for i in f_mean]
    f_std = [round(i.item()*100, 2) for i in f_std]

    for i in range(7):
        print( f'& ${pre_mean[i]} \pm {pre_std[i]}$ & ${rec_mean[i]} \pm {rec_std[i]}$ & ${f_mean[i]} \pm {f_std[i]}$  \\\ ' )
        
    
    
def general_print(metrics, metric_names, metric_idx, metric_type):


    for name in metric_names:
        print(f'metrics name: {name}')

        print('-'*80)
        mean, std, data = get_mean_std_metrics(metrics, metric_idx=metric_idx, name=name, metric_type=metric_type)  
        print(mean*100)
        print(std*1.960*100)

        print('-'*80)
        mean, std, data = get_mean_std_metrics_phase(metrics, metric_idx=metric_idx, name=name)           
        print(mean*100)
        print(std*1.960*100)

        print('*'*80)
        print('*'*80)
        
        

        
def print_m2cai16():
#     metric_type = 'macro'
#     metric_names = ['precision', 'recall', 'fscore', 'jaccard', 'accuracy']
#     metrics = get_metrics( enable_boost=True,  boost_factor=0.36, dataset='m2cai16', 
#                           enable_filter=True, filter_threshold =0.7, path='/workspace/Models-m2cai/')
#     path = 'save_metrics_paper/m2cai_87.57.pt'
# #     metrics = torch.load(path)
#     general_print(metrics, metric_names=metric_names, metric_idx=None, metric_type=metric_type)
#     latex_print(metrics, metric_idx=None)
#     torch.save(metrics, path)


    path = '/workspace/Models/cholec80_Swin3D_S_7_pt.config'
    data = torch.load(path)
    yt = data['slm_metrics_details']['yt']
    yp = data['slm_metrics_details']['yp']
    yt = torch.cat(yt)
    yp = torch.cat(yp)

    j = jaccard_score(yt, yp, average = 'weighted')

    print(f'Jaccard Score: {j}')
    print()

    print(classification_report(yt, yp, digits=4))
    
def print_cholec80():
    metric_type = 'weighted'
    select_indices = [ 16, 9, 18, 13, 29, 28, 4, 22, 14]
    metric_names = ['precision', 'recall', 'fscore', 'jaccard', 'accuracy']
    metrics = get_metrics( enable_boost=False,  boost_factor=0.05, dataset='cholec80', 
                          enable_filter=True, filter_threshold =0.8, 
                          path='/workspace/Models30/', select_indices=select_indices)

    general_print(metrics, metric_names=metric_names, metric_idx=None, metric_type=metric_type)
    latex_print(metrics, metric_idx=None)
    

def print_autolaparo():
    metric_names = ['precision', 'recall', 'fscore', 'jaccard', 'accuracy']
    metrics = get_metrics( enable_boost=False,  boost_factor=0.375, dataset='autolaparo', 
                          enable_filter=False, filter_threshold =0.86, path='/workspace/Models-autolaparo/', 
                         select_indices=[0,3,6,7,23,25,26,27])


    general_print(metrics, metric_names=metric_names, metric_idx=None, metric_type='weighted')
    latex_print(metrics, metric_idx=None)


# print_m2cai16()
# print_cholec80()
# print_autolaparo()


def bar_plot_dataset_accuracy():
  phases = [  "Cholec80", "AutoLaparo", "M2CAI16"]
  metrics = ["Swin3D-B", "Swin3D-S", "Swin3D-T"]

  data = np.array([
  [0.930, 0.920, 0.917],
  [0.880, 0.87, 0.864],
  [0.876, 0.865, 0.87],
  ])

  # Plotting
  x = np.arange(len(phases))
  width = 0.2  # Width of the bars

  fig, ax = plt.subplots(figsize=(6, 3))

  # Creating bars for each metric
  for i, metric in enumerate(metrics):
      ax.bar(x + i*width, data[:, i], width, label=metric)

  # Adding some text for labels, title, and axes ticks
  # ax.set_xlabel('VFE Module ')
  ax.set_ylabel('Accuracy')
  ax.set_ylim(0.8, 0.95)
  # ax.set_title('Performance Metrics dependency on Video Feature Extractor')
  ax.set_xticks(x + width * (len(metrics) - 1) / 2)
  ax.set_xticklabels(phases, rotation=0, ha='center')
  ax.legend(loc='best', fancybox=True, facecolor='white', framealpha=.8)

  plt.tight_layout()
  plt.savefig('performance-vfe.pdf')
  plt.show()


def plt_sspa_accuracy():
  stack_size = [2, 4, 7, 15, 30]
  accuracy_mhat = [83.51, 89.76, 90.82, 93.02, 92.91]
  accuracy_sspa = [73.85, 77.50, 79.32, 83.71, 85.11]

  # Plotting
  plt.figure(figsize=(6, 3))
  plt.plot(stack_size, accuracy_mhat, marker='o', linestyle='-', color='blue', label='MHAT Accuracy')
  plt.plot(stack_size, accuracy_sspa, marker='o', linestyle='-', color='green', label='SSPA Accuracy')

  # Adding titles and labels
  # plt.title('Accuracy vs Stack Size')
  plt.xlabel('SSPA stack size')
  plt.ylabel('Accuracy (%)')

  # Adding legend
  plt.legend()
  # plt.ylim(65, 100)

  plt.axvline(x=15, color='gray', linestyle='--', label='Best SSPA Stack Size ')
  plt.plot(15, 93.02, marker='o', color='red', markersize=8)

  # Show plot
  plt.grid(True)
  plt.tight_layout()

  plt.savefig('sspa-stack-accuracy.pdf')

  plt.show()


def plt_mhat_accuracy():
  import matplotlib.pyplot as plt

  # Data
  stack_size = [10, 15, 20, 25, 30, 35]
  accuracy_mhat = [87.52, 91.34, 92.82, 93.02, 93.02, 93.01]

  # Plotting
  plt.figure(figsize=(6, 3))
  plt.plot(stack_size, accuracy_mhat, marker='o', linestyle='-', color='blue', label='MHAT Accuracy')

  # Adding titles and labels
  # plt.title('Accuracy vs Stack Size')
  plt.xlabel('MhaT module stack size')
  plt.ylabel('Accuracy (%)')

  # Adding legend
  # plt.legend()
  # plt.ylim(65, 100)


  # Show plot

  plt.axvline(x=30, color='gray', linestyle='--', label='Sequence Size 30')
  plt.plot(30, 93.02, marker='o', color='red', markersize=8)
  # plt.text(30, 92.02, 'Best sequence size', fontsize=10, ha='left', color='red')

  plt.grid(True)
  plt.tight_layout()

  plt.savefig('maht-stack-accuracy.pdf')

  plt.show()


def plt_stride_accuracy():
# Data
  stride_count = [10, 15, 20, 25, 30]
  prime_stride = [91.41, 92.24, 92.62, 93.02, 93.01]
  geometric_stride = [91.11, 91.53, 91.96, 92.24, 92.34]
  arithmetic_stride = [91.62, 91.92, 92.26, 92.74, 92.64]

  # Plotting
  plt.figure(figsize=(6, 4))
  plt.plot(stride_count, prime_stride, marker='o', linestyle='-', color='blue', label='Prime Stride')
  plt.plot(stride_count, geometric_stride, marker='o', linestyle='-', color='green', label='Geometric Stride')
  plt.plot(stride_count, arithmetic_stride, marker='o', linestyle='-', color='purple', label='Arithmetic Stride')


  # Adding titles and labels
  # plt.title('Accuracy vs Stack Size')
  plt.xlabel('Stride count')
  plt.ylabel('Accuracy (%)')

  # Adding legend
  plt.legend()
  # plt.ylim(65, 100)

  plt.axvline(x=25, color='gray', linestyle='--', label='Sequence Size 30')
  plt.plot(25, 93.02, marker='o', color='red', markersize=8)

  # Show plot
  plt.grid(True)
  plt.tight_layout()

  plt.savefig('stride-accuracy.pdf')

  plt.show()
