import os
import fnmatch
import argparse
import numpy as np

def mkfp(*dirs):
    return os.path.join('/home/lab149/Multi-Stage-LSTM-for-Action-Anticipation-master/data/FPS100/', *dirs)

total_frames = 0
total_num = 1
for i in range(1,121):
    #print(cat)
    
    cat_odir = os.path.join('/home/lab149/Multi-Stage-LSTM-for-Action-Anticipation-master/data/FPS100_3/', 'train')
    if not os.path.isdir(cat_odir):
        os.makedirs(cat_odir)
        
    if i%30 <= 10 and i%30 >= 1:
        cat_odir_ = os.path.join(cat_odir,'jiandao')
    elif i%30 <= 20:
        cat_odir_ = os.path.join(cat_odir,'shitou')
    else:
        cat_odir_ = os.path.join(cat_odir,'bu')
    

    for vid in fnmatch.filter(os.listdir(mkfp(i)), "*.jpg"):
        vpath = mkfp(i, vid)

        os.symlink(vpath,os.path.join(cat_odir, i+'_'+vid))
        total_num = total_num+1

print("Total frames decoded: %d" % total_frames)
