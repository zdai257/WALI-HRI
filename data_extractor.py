#!/usr/bin/env python

import os
from os.path import join
from glob import glob
import shutil
import subprocess


dataset_path = r"/media/daiz1/Seagate Basic/RECORDINGS"
master_fd = "MASTER_PUPIL_SOUND"
sub_fd = "SUB1"

target_path = "/home/CAMPUS/daiz1/repo/WALI-HRI/dataset"
target_fd = "GBSIOT_02B"



for fd_name in os.listdir(join(target_path)):
    if fd_name == target_fd and not fd_name.startswith("."):
        #print(fd_name)
        #print(join(dataset_path, master_fd))
        #print(glob("./**/*.wav"))
        for dates in os.listdir(join(dataset_path, master_fd)):
        #for item in os.listdir(r"/media/daiz1/Seagate Basic/RECORDINGS/MASTER_PUPIL_SOUND/2023_07_13/GBSIOT_02A"):
            if os.path.exists(join(dataset_path, master_fd, dates, target_fd)):
                src_fd = join(dataset_path, master_fd, dates, target_fd)
                for item in os.listdir(src_fd):
                
                    # Sound
                    if item.endswith('.wav'):
                        src = join(src_fd, item)
                        dst = join(target_path, target_fd, 'Sound')
                        cmd = f'cp "{src}" "{dst}"'
                        os.system(cmd)
                        
                    # Pupil
                    if item.endswith('_annotated'):
                        for root, dirs, files in os.walk(join(src_fd, item), topdown=False):
                            for filename in files:
                                if filename == "info.player.json":
                                    src = join(src_fd, item, root, filename)
                                    dst = join(target_path, target_fd, 'Pupil')
                                    
                                    
                    # MKV
                    if item.endswith('.mkv'):
                        pass
                                
            
                
                
                






#if not os.path.exists(join("/home/CAMPUS/daiz1/Pictures", vid_idx)):
#    os.makedirs(join("/home/CAMPUS/daiz1/Pictures", vid_idx))

#os.system("ffmpeg -i {0} -f image2 -vf fps=fps=1 /home/CAMPUS/daiz1/Pictures/output%d.png".format(vid_path))
#os.system("ffmpeg -i {0} -vf \"select='eq(pict_type,I)'\" -vsync vfr /home/CAMPUS/daiz1/Pictures/{1}/out-%02d.jpeg".format(vid_path, vid_idx))
