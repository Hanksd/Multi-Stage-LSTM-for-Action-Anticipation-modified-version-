import argparse
import glob
import json
import cv2
import numpy as np
import os
from models import vgg_context
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
    "--model",
    required=True,
    type=str,
    default='model_weights/context_aware_vgg16_final.h5',
    help="path to the trained model of context_aware")

parser.add_argument(
    "--split-dir",
    type=str,
    default='model_weights/context_aware_vgg16_final.h5',
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

data_dir = 'data/jhmdb_dataset/'
split_dir = 'data/splits/'
classes_ = 21
model_ = 'data/model_weights/context_best.h5'
temporal_length = 40
split_ = 1
output = 'data/context_features/'
fixed_width = 224


model = vgg_context(classes_, input_shape=(fixed_width,fixed_width,3))
model.load_weights(model_)

context_aware = K.function([model.layers[0].input, K.learning_phase()], [model.layers[22].output])

data_mean = json.load(open('config/mean.json', 'r'))

classes = [ d for d in os.listdir(data_dir) ]
classes = list(filter(lambda d: os.path.isdir(os.path.join(data_dir, d)), classes))
classes = sorted(classes)
print(classes)

for cls in sorted(classes):
    with open(os.path.join(split_dir, cls + '_test_split%d.txt' % split_)) as f:
        lines = f.readlines()

    for l in lines:
        video_fn, split = l.split()

        if split == '1':
            cat = 'train'
        elif split == '2':
            cat = 'val'
        else:
            cat = 'dummy'

        print (video_fn, split, cls)

        feature = np.zeros((temporal_length,4096))
        label = np.zeros((temporal_length,len(classes)))

        vid_path = os.path.join(data_dir, cls)
        cap = cv2.VideoCapture(os.path.join(vid_path, video_fn))

        for fr in range(temporal_length):
            label[fr][classes.index(cls)] = 1

            try:
                ret,frame = cap.read()
                if ret:
                    cv2.imwrite('data/1.png', frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
                    f2 = load_img('data/1.png',grayscale=False, target_size=(224,224),interpolation='nearest')
                    f2_arr = np.array(f2, dtype=np.float32)
                    #f2 = cv2.resize(frame, (fixed_width,fixed_width), interpolation=cv2.INTER_CUBIC)
                    #f2_arr = np.array(f2, dtype=np.float64)

                    f2_arr[:, :, 0] -= data_mean[0]
                    f2_arr[:, :, 1] -= data_mean[1]
                    f2_arr[:, :, 2] -= data_mean[2]

                    in_ = np.expand_dims(f2_arr, axis=0)

                    feature[fr] = np.array(context_aware([in_, 0]))[0][0]
                    #label[fr][classes.index(cls)] = 1
                    print('ok')

            except:
                pass

        np.save(os.path.join(output, cat+'/feature_'+video_fn.split('.')[0]+'.npy'), feature)
        np.save(os.path.join(output, cat+'/label_' + video_fn.split('.')[0] + '.npy'), label)

        print ("[Done] " + video_fn + " " + cls)