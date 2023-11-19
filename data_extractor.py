#!/usr/bin/env python

import os
from os.path import join
from glob import glob
import shutil
import subprocess
import re


dataset_path = r"/media/daiz1/Seagate Basic/RECORDINGS"
master_fd = "MASTER_PUPIL_SOUND"
sub_fd = "SUB1"

target_path = "/home/CAMPUS/daiz1/repo/WALI-HRI/dataset"
target_fd = "GBSIOT_02A"


if True:
    for fd_name in os.listdir(join(target_path)):
        # Constrain to a folder
        #if fd_name == target_fd and not fd_name.startswith("."):
        if not fd_name.startswith("."):
        
            #print(join(dataset_path, master_fd))
        
            for dates in os.listdir(join(dataset_path, master_fd)):
        
                if os.path.exists(join(dataset_path, master_fd, dates, fd_name)):
                    src_fd = join(dataset_path, master_fd, dates, fd_name)
                    for item in os.listdir(src_fd):
                
                        # Sound
                        if item.endswith('.wav'):
                            src = join(src_fd, item)
                            dst = join(target_path, fd_name, 'Sound')
                            cmd = f'cp "{src}" "{dst}"'
                            os.system(cmd)
                        
                        # Pupil
                        if item.endswith('_annotated'):
                            for root, dirs, files in os.walk(join(src_fd, item), topdown=False):
                                for filename in files:
                                    if filename == "info.player.json":
                                        src = join(src_fd, item, root, filename)
                                        dst = join(target_path, fd_name, 'Pupil')
                                        #print(src, dst)
                                        cmd = f'cp "{src}" "{dst}"'
                                        os.system(cmd)
                                    
                                    if re.match(filename, r"eye[0-9]_timestamps.npy"):
                                        src = join(src_fd, item, root, filename)
                                        dst = join(target_path, fd_name, 'Pupil')
                                        os.system(f'cp "{src}" "{dst}"')
                                    
                                    if filename == "gaze_positions.csv":
                                        src = join(src_fd, item, root, filename)
                                        dst = join(target_path, fd_name, 'Pupil')
                                        os.system(f'cp "{src}" "{dst}"')
                                    
                                    if filename == "blinks.csv":
                                        src = join(src_fd, item, root, filename)
                                        dst = join(target_path, fd_name, 'Pupil')
                                        os.system(f'cp "{src}" "{dst}"')
                                    
                                    if filename == "fixations.csv":
                                        src = join(src_fd, item, root, filename)
                                        dst = join(target_path, fd_name, 'Pupil')
                                        os.system(f'cp "{src}" "{dst}"')
                                    
                                    if filename == "head_pose_tracker_poses.csv":
                                        src = join(src_fd, item, root, filename)
                                        dst = join(target_path, fd_name, 'Pupil')
                                        os.system(f'cp "{src}" "{dst}"')
                                    
                                    if filename == "world_timestamps.csv":
                                        src = join(src_fd, item, root, filename)
                                        dst = join(target_path, fd_name, 'Pupil')
                                        os.system(f'cp "{src}" "{dst}"')
                                    
                                    if filename == "world.mp4" and 'exports' in src:
                                        src = join(src_fd, item, root, filename)
                                        dst = join(target_path, fd_name, 'Pupil')
                                        dst_worldfile = dst + '/world_overlay.mp4'
                                        print(src, dst_worldfile)
                                        os.system(f'cp "{src}" "{dst_worldfile}"')
                                    
                                    # with eye overlay
                                    if filename == "world.mp4" and 'exports' not in src:
                                        src = join(src_fd, item, root, filename)
                                        dst = join(target_path, fd_name, 'Pupil')
                                        print(src, dst)
                                        os.system(f'cp "{src}" "{dst}"')
                                    
                                    
                        # MKV master
                        if item.endswith('.mkv'):
                            # Depth
                            src = join(src_fd, item)
                            dst = join(target_path, fd_name, 'Azure_3rd', 'Depth.mp4')
                            cmd = f'ffmpeg -i "{src}" -map 0:1 -vsync 0 "{dst}"'
                            os.system(cmd)
                            # IR
                            dst = join(target_path, fd_name, 'Azure_3rd', 'IR.mp4')
                            cmd = f'ffmpeg -i "{src}" -map 0:2 -vsync 0 "{dst}"'
                            os.system(cmd)
                            ### exclude ###
                            # YUV
                            dst = join(target_path, fd_name, 'Azure_3rd', 'YUV.mp4')
                            cmd = f'ffmpeg -i "{src}" -map 0:0 -vsync 0 -pix_fmt yuv420p "{dst}"'
                            os.system(cmd)
                    
                    
                    # MKV sub
                    src_fd2 = join(dataset_path, sub_fd, dates, fd_name)
                    for item in os.listdir(src_fd2):
                        if item == 'sub1.mkv':
                            # Depth
                            src = join(src_fd2, item)
                            dst = join(target_path, fd_name, 'Azure_top', 'Depth.mp4')
                            cmd = f'ffmpeg -i "{src}" -map 0:1 -vsync 0 "{dst}"'
                            os.system(cmd)
                            # IR
                            dst = join(target_path, fd_name, 'Azure_top', 'IR.mp4')
                            cmd = f'ffmpeg -i "{src}" -map 0:2 -vsync 0 "{dst}"'
                            os.system(cmd)
                            # YUV
                            dst = join(target_path, fd_name, 'Azure_top', 'YUV.mp4')
                            cmd = f'ffmpeg -i "{src}" -map 0:0 -vsync 0 -pix_fmt yuv420p "{dst}"'
                            os.system(cmd)
    
        print("Done folder {0}\n".format(fd_name))
                                
            

