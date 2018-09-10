import os
import cv2
import subprocess
import numpy as np
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

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def cv2_dump_frames(total_n,fn, output_path,fmt="jpg", quality=90):

    cap = cv2.VideoCapture(fn)
    
    a = int(total_n)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    index = 0
    #ay = np.zeros((224,224,3))
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        index += 1
        
        print(index)

        if fmt == "webp":
            cv2_ext = "png"
        else:
            cv2_ext = fmt

        fn = os.path.join(output_path, '%08d.%s' % (index, cv2_ext))
        cv2.imwrite(fn, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if index == 1:
            f2 = load_img(fn,grayscale=False, target_size=(224,224),interpolation='nearest')
            f2_arr = np.array(f2, dtype=np.float32)
            f2.save(os.path.join(output_path2, '%08d.%s' % (a, cv2_ext)))
        if fmt == 'webp':
            with open("/dev/null", "w") as null:
                subprocess.check_call(['cwebp', fn, '-lossless', '-noalpha', '-mt', '-o', os.path.splitext(fn)[0] + ".webp"], stdout=null, stderr=null)
                os.remove(fn)
    print(index)

    return index,f2_arr

'''
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='convert torch lmdb to portable lmdb')

    parser.add_argument(
        "--video",
        metavar="<path>",
        required=True, type=str,
        help="input video")

    parser.add_argument(
        "--output-dir",
        metavar="<path>",
        required=True,
        type=str,
        help="output base directory")

    parser.add_argument(
        "--format",
        metavar="<path>",
        default='jpg',
        choices=['jpg', 'png', 'webp'],
        type=str,
        help="output image format")

    args = parser.parse_args()
    cv2_dump_frames(args.video, args.output_dir, args.format, 94)
'''
'''
cv2_dump_frames('./data/jhmdb_dataset/wave/50_FIRST_DATES_wave_u_cm_np1_fr_goo_30.avi', 
                './data/frames/wave/50_FIRST_DATES_wave_u_cm_np1_fr_goo_30.avi', 
                'png', 94)
'''
