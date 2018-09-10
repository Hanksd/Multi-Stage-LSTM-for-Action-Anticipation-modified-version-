from models import vgg_action, vgg_context
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import argparse
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from keras.applications.vgg16 import preprocess_input

'''
parser = argparse.ArgumentParser(description='tune vgg16 network on new dataset')

parser.add_argument(
    "--data-dir",
    metavar="<path>",
    required=True,
    type=str,
    help="train/val data base directory")

parser.add_argument(
    "--classes",
    type=int,
    default=21,
    help="number of classes in target dataset")

parser.add_argument(
    "--model-type",
    choices=['action_aware', 'context_aware'],
    default='context_aware',
    help="action-aware model or context-aware model")

parser.add_argument(
    "--epochs",
    default=128,
    type=int,
    help="number of epochs")

parser.add_argument(
    "--samples-per-epoch",
    default=None,
    type=int,
    help="samples per epoch, default=all")

parser.add_argument(
    "--save-model",
    metavar="<prefix>",
    default=None,
    type=str,
    help="save model at the end of each epoch")

parser.add_argument(
    "--save-best-only",
    default=False,
    action='store_true',
    help="only save model if it is the best so far")

parser.add_argument(
    "--num-val-samples",
    default=None,
    type=int,
    help="number of validation samples to use (default=all)")

parser.add_argument(
    "--fixed-width",
    default=224,
    type=int,
    help="crop or pad input images to ensure given width")

parser.add_argument(
    "--seed",
    default=10,
    type=int,
    help="random seed")

parser.add_argument(
    "--workers",
    default=1,
    type=int,
    help="number of data preprocessing worker threads to launch")

parser.add_argument(
    "--learning-rate",
    default=0.001,
    type=float,
    help="initial/fixed learning rate")

parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="batch size")


args = parser.parse_args()
'''
def preprocess_img(img):
    data_mean = [102.79, 103.38, 90.04]
    img = np.array(img)
    img[:, :, 0] -= data_mean[0]
    img[:, :, 1] -= data_mean[1]
    img[:, :, 2] -= data_mean[2]
    return img

correct_model = False
model_type='action_aware'
classes = 21
fixed_width = 224
batch_size = 32
learning_rate = 0.001
save_model = 'data/model_weights/action_best.h5'
samples_per_epoch = None
epochs = 128
workers = 1
num_val_samples = None

if model_type == 'action_aware':
    model = vgg_action(classes, input_shape=(fixed_width,fixed_width,3))
    correct_model = True
elif model_type == 'context_aware':
    model = vgg_context(classes, input_shape=(fixed_width, fixed_width, 3))
    correct_model = True
else:
    print("Wrong model type name!")

if correct_model:
    '''

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    '''
    
    test_datagen = ImageDataGenerator(
        #rescale=1./255
        #preprocessing_function=preprocess_input
        featurewise_center=True,
        #samplewise_center=True,
        featurewise_std_normalization=True
        #samplewise_std_normalization=True
        )

    train_datagen = ImageDataGenerator(
        #rescale=1./255,
        #preprocessing_function=preprocess_input,
        featurewise_center=True,
        #samplewise_center=True,
        featurewise_std_normalization=True,
        #samplewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        channel_shift_range = 0.3,
        horizontal_flip=True)
    
    images = np.load('/home/lab149/Multi-Stage-LSTM-for-Action-Anticipation-master/data/1.npy')
    test_datagen.fit(images)
    train_datagen.fit(images)


    train_generator = train_datagen.flow_from_directory(
            os.path.join('data/splitted_data/' , 'train/'),
            target_size=(fixed_width, fixed_width),
            batch_size=batch_size,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data/splitted_data/', 'val/'),
            target_size=(fixed_width, fixed_width),
            batch_size=batch_size,
            class_mode='categorical')


    sgd = SGD(lr=learning_rate, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = []

    if save_model:
        callbacks.append(ModelCheckpoint(save_model,
                                         verbose=0,
                                         monitor='val_acc',
                                         save_best_only=True))

    samples_per_epoch = samples_per_epoch or train_generator.samples // batch_size
    samples_per_epoch -= (samples_per_epoch % batch_size)
    num_val_samples = num_val_samples or validation_generator.samples // batch_size


    print("Starting to train...")
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        verbose=1,
                        callbacks=callbacks,
                        epochs=epochs,
                        workers=workers,
                        shuffle=True,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples // batch_size)

    if model_type == 'action_aware':
        model.save_weights('data/model_weights/action_aware_vgg16_final.h5')
    else:
        model.save_weights('data/model_weights/context_aware_vgg16_final.h5')



