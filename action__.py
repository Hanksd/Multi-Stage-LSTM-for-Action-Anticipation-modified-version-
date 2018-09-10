import argparse
import glob
import json
import cv2
import numpy as np
import os
from models import vgg_action, vgg_context
np.set_printoptions(threshold=np.inf)
from keras import backend as K
try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
        
output_path2 = '/home/lab149/Multi-Stage-LSTM-for-Action-Anticipation-master/data/2'
        
def load_img(path, grayscale=False, target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img

'''
parser = argparse.ArgumentParser(description='extracting context-aware features')

parser.add_argument(
    "--data-dir",
    metavar="<path>",
    required=True,
    type=str,
    default='data/jhmdb_dataset/',
    help="path to video files")

parser.add_argument(
    "--classes",
    type=int,
    default=21,
    help="number of classes in target dataset")

parser.add_argument(
    "--model-action",
    required=True,
    type=str,
    default='model_weights/action_aware_vgg16.h5',
    help="path to the trained model of action_aware")

parser.add_argument(
    "--model-context",
    required=True,
    type=str,
    default='model_weights/context_aware_vgg16.h5',
    help="path to the trained model of context_aware")

parser.add_argument(
    "--split-dir",
    type=str,
    default='model_weights/context_aware_vgg16.h5',
    help="path to the dataset splits directory")

parser.add_argument(
    "--temporal-length",
    default=50,
    type=int,
    help="number of frames representing each video")

parser.add_argument(
    "--split",
    default='1',
    type=str,
    help="the split")

parser.add_argument(
    "--output",
    default='data/context_features/',
    type=str,
    help="path to the directory of features")

parser.add_argument(
    "--fixed-width",
    default=224,
    type=int,
    help="crop or pad input images to ensure given width")


args = parser.parse_args()
'''



data_dir_ = 'data/jhmdb_dataset/'
split_dir_ = 'data/splits/'
classes_ = 21
model_action_ = 'data/model_weights/action_best.h5'
model_context_ = 'data/model_weights/context_best.h5'
temporal_length_ = 40
split_ = 1
output_ = 'data/action_features/'
fixed_width_ = 224


model_action = vgg_action(classes_, input_shape=(fixed_width_,fixed_width_,3))
model_action.load_weights(model_action_)
model_action.summary()

model_context = vgg_context(classes_, input_shape=(fixed_width_,fixed_width_,3))
model_context.load_weights(model_context_)
model_context.summary()


context_aware = K.function([model_context.layers[18].input, K.learning_phase()], [model_context.layers[22].output])
context_conv = K.function([model_context.layers[0].input, K.learning_phase()], [model_context.layers[17].output])
cam_conv = K.function([model_action.layers[0].input, K.learning_phase()], [model_action.layers[19].output])
cam_fc = model_action.layers[-1].get_weights()
action_aware = K.function([model_action.layers[18].input, K.learning_phase()], [model_action.layers[22].output])


data_mean = json.load(open('config/mean.json', 'r'))
#print(data_mean)
'''
cap = cv2.VideoCapture('/home/lab149/Multi-Stage-LSTM-for-Action-Anticipation-master/data/jhmdb_dataset/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi')
ret,frame = cap.read()
f2 = cv2.resize(frame, (fixed_width_,fixed_width_), interpolation=cv2.INTER_CUBIC)
f2_arr = np.array(f2, dtype=np.float64)
f2_arr[:, :, 0] -= data_mean[0]
f2_arr[:, :, 1] -= data_mean[1]
f2_arr[:, :, 2] -= data_mean[2]
in_ = np.expand_dims(f2_arr, axis=0)
CONV5_out = np.array(context_conv([in_, 0]))[0]
print(CONV5_out.shape)

'''
classes = [ d for d in os.listdir(data_dir_) ]
classes = list(filter(lambda d: os.path.isdir(os.path.join(data_dir_, d)), classes))
#print(sorted(classes))
classes = sorted(classes)
#for cls in sorted(classes):
    #print(classes.index(cls))

for cls in sorted(classes):
    with open(os.path.join(split_dir_, cls + '_test_split%d.txt' % split_)) as f:
        lines = f.readlines()

    for l in lines:
        video_fn, split = l.split()

        if split == '1':
            cat = 'train'
            continue
        elif split == '2':
            cat = 'val'
        else:
            cat = 'dummy'

        print (video_fn, split, cls)

        feature = np.zeros((temporal_length_,4096))
        label = np.zeros((temporal_length_,len(classes)))

        vid_path = os.path.join(data_dir_, cls)
        t_path = os.path.join(vid_path, video_fn)
        print(t_path)
        cap = cv2.VideoCapture(os.path.join(vid_path, video_fn))

        for fr in range(temporal_length_):
            #print(classes.index(cls))
            
            label[fr][classes.index(cls)] = 1

            try:
                ret,frame = cap.read()
                #print('ok')
                if ret:
                    cv2.imwrite('data/1.png', frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
                    f2 = load_img('data/1.png',grayscale=False, target_size=(224,224),interpolation='nearest')
                    f2_arr = np.array(f2, dtype=np.float32)
                    #f2 = cv2.resize(frame, (fixed_width_,fixed_width_), interpolation=cv2.INTER_CUBIC)
                    #f2_arr = np.array(f2, dtype=np.float64)

                    f2_arr[:, :, 0] -= data_mean[0]
                    #print(data_mean[0])
                    f2_arr[:, :, 1] -= data_mean[1]
                    f2_arr[:, :, 2] -= data_mean[2]
                    #print('ok')
                    

                    in_ = np.expand_dims(f2_arr, axis=0)
                    
                    f_ = model_action.predict([in_])
                    f_f = np.argsort(f_)[0,-3:]
                    
                    #print(f_.shape)
                    #print(f_f)

                    CONV5_out = np.array(context_conv([in_, 0]))[0]

                    cam_fc = model_action.layers[-1].get_weights()

                    CAM_conv = np.array(cam_conv([in_, 0]))[0]
                    
                    #print('ok')
                    S = np.zeros((3,14, 14))
                    for b in range(3):
                        #print(b)
                        for j in range(1024):#1024— 4096
                            S[b] = S[b] + (cam_fc[0][j][f_f[b]] * CAM_conv[0, :, :, j])
                        #S[b] = S[b]*f_[0][f_f[b]]
                    S = (abs(S)+S)/2.
                    #print('ok')
                    #print(S)
                    #print(classes.index(cls))
                    #S = S.reshape((4116))
                    #S = S/np.linalg.norm(S)
                    #S = S.reshape((21,14,14))
                    S = np.sum(S,axis=0)
                    #S = S*1./3.
                    #print(S.shape)
                    #SS = S
                    #SS = S/np.amax(S)

                    SS = (S - np.min(S)) / (np.max(S) - np.min(S))
                    #print(SS)
                    #SS = (abs(S)+S)/2.
                    #print(SS)
                    feat_inp = np.zeros((1, 14, 14, 512))
                    #print('ok')
                    for i in range(0, 512):
                        feat_inp[0, :, :, i] = CONV5_out[0, :, :, i] * SS
                    feat_inp = (feat_inp / np.mean(feat_inp)) * np.mean(CONV5_out)
                    #print('ok')
                    #print(feat_inp.shape)
                    #a = np.array(action_aware([feat_inp, 0]))[0][0]
                    #print(a.shape)

                    feature[fr] = np.array(context_aware([feat_inp, 0]))[0][0]#action_aware -> context_aware
                    #print('ok')
                    #label[fr][classes.index(cls)] = 1
                    print('ok')
                    #break

            except:
                pass

        np.save(os.path.join(output_, cat+'/feature_'+video_fn.split('.')[0]+'.npy'), feature)
        np.save(os.path.join(output_, cat+'/label_' + video_fn.split('.')[0] + '.npy'), label)

        print ("[Done] " + video_fn + " " + cls)
        #break
    #break
