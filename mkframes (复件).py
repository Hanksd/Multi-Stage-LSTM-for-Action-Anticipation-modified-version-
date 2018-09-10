import os
import fnmatch
import argparse
import numpy as np
import make_frames
'''
parser = argparse.ArgumentParser(description='converting videos to frames')

parser.add_argument(
    "--input-dir",
    metavar="<path>",
    default='data/jhmdb_dataset/',
    required=True,
    type=str,
    help="base directory of classes")

parser.add_argument(
    "--output-dir",
    metavar="<path>",
    defult='data/frames/',
    required=True,
    type=str,
    help="output base dir")

parser.add_argument(
    "--format",
    metavar="<path>",
    default='png',
    choices=['jpg', 'png', 'webp'],
    type=str,
    help="output image format")

args = parser.parse_args()
'''
def mkfp(*dirs):
    return os.path.join('data/jhmdb_dataset/', *dirs)

total_frames = 0
fr_t = []
total_num = 1
for cat in sorted(os.listdir('data/jhmdb_dataset/')):
    print(cat)
    
    cat_odir = os.path.join('data/frames/', cat)
    if not os.path.isdir(cat_odir):
        os.makedirs(cat_odir)

    for vid in fnmatch.filter(os.listdir(mkfp(cat)), "*.avi"):
        vpath = mkfp(cat, vid)

        odir = os.path.join(cat_odir, vid)
        if not os.path.isdir(odir):
            os.mkdir(odir)

        print("Decoding '%s' '%s'" % (vpath,odir))
        num_frames,fr =  make_frames.cv2_dump_frames(total_num, vpath, odir, "png", 94)
        total_num = total_num+1
        total_frames += num_frames
        fr_t.append(fr)
        
ffr = np.array(fr_t)
print(ffr.shape)
np.save(os.path.join('/home/lab149/Multi-Stage-LSTM-for-Action-Anticipation-master/data', '1.npy'), ffr)

print("Total frames decoded: %d" % total_frames)



