import os
import torchvision

'''
A simple program to download files with file ids given in the list. 
'''

file_ids = []
dataset_path = 'Dataset/Cholec80/96x96@5fps'
os.makedirs(dataset_path, exist_ok=True)

for i, file_id in enumerate(file_ids):
  file_name = 'video{:02d}.tfrecord'.format(i+1)
  print(file_name)
  torchvision.datasets.utils.download_file_from_google_drive(file_id=file_id, root=dataset_path, filename=file_name)
  break
