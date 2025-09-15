#!/usr/bin/python
# 複製相同的label
import os
import shutil

color_t = 'Zsamples'
source_path = 'temp/'+color_t+'_416/'
target_path = 'temp/'+color_t+'_label/'

img_files = os.listdir(source_path)
source_file = img_files[0][0:len(img_files[0])-3] + 'txt'

for cnt in range(1, len(img_files)):
    label_file = img_files[cnt][0:len(img_files[cnt])-3] + 'txt'
    shutil.copyfile(target_path+source_file, target_path+label_file)

print('Done!')

