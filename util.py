import os,sys
import numpy as np


def organize_result(inpath='/media/chaitanya/DATADRIVE0/github/physim-dataset_generator-data/cycles_jdx/rendered_images_4/',outpath='/media/chaitanya/DATADRIVE0/github/physim-dataset_generator-data/cycles_jdx/training_data/'):
  
  num_training_images = 10000

  if not os.path.exists(inpath):
    print('inpath is null!')
  
  src=['rgb/*', 'labels/edge_img.png', 'labels/seg_img.png', 'depth/*']
  dst=['rgb/','edge_labels/','seg_labels/','depth/']

  for item in dst:
    if not os.path.exists(outpath+item):
      os.makedirs(outpath+item)

  cnt = 10224

  for scene_id in range(0, num_training_images):

    for i in range(len(src)):
      if 'depth' in src[i]:
        command ='cp ' + inpath + 'image_%05d' % scene_id + '/' + src[i] + ' ' + outpath + dst[i] + '%06d' % cnt + '.exr'
      else:
        command ='cp ' + inpath + 'image_%05d' % scene_id + '/' + src[i] + ' ' + outpath + dst[i] + '%06d' % cnt + '.png'

      print(command)
      res = os.system(command)
      if (res != 0) :
        print('error:\n',command)
        exit(1)
    cnt += 1

def renameDepth(inpath='./rendered_images2/depth/'):
  files = os.listdir(inpath)
  
  for file in files:
    if 'png' in file:
      name = os.path.splitext(file)[0]
      cmd = 'mv '+inpath+file+' '+inpath+name+'.exr'
      print(cmd)
      os.system(cmd)



if __name__=='__main__':
  organize_result()
  # renameDepth()