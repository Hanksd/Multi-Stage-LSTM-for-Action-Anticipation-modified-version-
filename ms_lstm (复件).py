from models import MS_LSTM
import numpy as np
#from theano.tensor import basic as tensor
from keras import backend as K
from keras.optimizers import SGD
import argparse
from keras.callbacks import ModelCheckpoint
from feature_generator import CustomDataGenerator
'''
parser = argparse.ArgumentParser(description='Train multi-stage LSTM (MS-LSTM)')

parser.add_argument(
    "--action-aware",
    metavar="<path>",
    required=True,
    type=str,
    help="path to action-aware features")

parser.add_argument(
    "--context-aware",
    metavar="<path>",
    required=True,
    type=str,
    help="path to context-aware features")

parser.add_argument(
    "--classes",
    type=int,
    default=21,
    help="number of classes in target dataset")

parser.add_argument(
    "--loss-type",
    choices=['crossentropy', 'hinge', 'totally_linear', 'partially_linear', 'exponential'],
    default='crossentropy',
    help="The loss function to train MS-LSTM")

parser.add_argument(
    "--epochs",
    default=128,
    type=int,
    elp="number of epochs")

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

parser.add_argument(
    "--temporal-length",
    default=50,
    type=int,
    help="number of frames representing each video")


parser.add_argument(
    "--cell",
    default=2048,
    type=int,
    help="number of hidden units in LSTM cells")

args = parser.parse_args()
'''

action_aware_ = 'data/action_features/'
context_aware_ = 'data/context_features/'
classes_ = 21
epochs_ = 128
save_model_ = 'data/model_weights/mslstm_best.h5'
learning_rate_ = 0.001
batch_size_ = 32
temporal_length_ = 40
cell_ =  2048
save_best_only_ = False
#loss_type_ = 'crossentropy'
loss_ = 'partially_linear'
samples_per_epoch_ = None
num_val_samples_ = None
workers_ = 1

def totally_linear(y_true, y_pred):
        exp_loss = 0
        T = 18
        for t in range(0,21):
                exp_loss = exp_loss + ((np.float64(t)/(T)) * (K.categorical_crossentropy(y_pred, y_true)))

        return exp_loss


def totally_expontial(y_true, y_pred):
    exp_loss = 0
    T = 18
    for t in range(0, 21):
        exp_loss = exp_loss + (np.exp((-1) * (T - t)) * K.categorical_crossentropy(y_pred, y_true))

    return exp_loss

'''
def partially_linear(true_dist, coding_dist):
        loss = 0
        TIME = 150
        N_C = 21
        batch = 32
        for t in range (TIME):
                term1 = true_dist[:,t] * tensor.log(coding_dist[:,t]+0.0000001)
                term2 = (1-true_dist[:,t]) * tensor.log(1-coding_dist[:,t]+0.0000001)
                loss = loss + np.double(1)/N_C * tensor.sum(term1+term2*np.double(t)/TIME, axis=1)

        return -loss/batch
'''

def partially_linear(true_dist, coding_dist):
        loss = 0
        TIME = 40
        N_C = 21
        batch = 32
        for t in range (TIME):
                term1 = true_dist[:,t] * K.log(coding_dist[:,t]+0.0000001)
                term2 = (1-true_dist[:,t]) * K.log(1-coding_dist[:,t]+0.0000001)
                loss = loss + np.float64(1)/N_C * K.sum(term1+term2*np.float64(t)/TIME, axis=1)

        return -loss/batch

def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)

    return K.maximum(0., neg - pos + 1.)


def categorical_crossentropy(y_true, y_pred):

    return K.categorical_crossentropy(y_true, y_pred)


model = MS_LSTM(INPUT_LEN=temporal_length_, INPUT_DIM=4096, OUTPUT_LEN=classes_, cells=cell_)

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

if loss_ == "crossentropy": model.compile(
        loss={'stage1':'categorical_crossentropy', 'stage2':'categorical_crossentropy'},
        optimizer=sgd, metrics=['accuracy'])

elif loss_ == "hinge": model.compile(
        loss={'stage1': categorical_hinge, 'stage2': categorical_hinge},
        optimizer=sgd, metrics=['accuracy'])

elif loss_ == "totally_linear": model.compile(
        loss={'stage1': totally_linear, 'stage2': totally_linear},
        optimizer=sgd, metrics=['accuracy'])

elif loss_ == "partially_linear": model.compile(
        loss={'stage1': partially_linear, 'stage2': partially_linear},
        optimizer=sgd, metrics=['accuracy'])

elif loss_ == "exponential": model.compile(
        loss={'stage1': totally_expontial, 'stage2': totally_expontial},
        optimizer=sgd, metrics=['accuracy'])

else: model.compile(
        loss={'stage1':'categorical_crossentropy', 'stage2':'categorical_crossentropy'},
        optimizer=sgd, metrics=['accuracy'])


callbacks = []

if save_model_:
    callbacks.append(ModelCheckpoint(save_model_,
                                         verbose=0,
                                         monitor='val_stage1_acc',
                                         save_best_only=True))


train_generator_obj = CustomDataGenerator(
    data_path_context=context_aware_ + '/train/',
    data_path_action=action_aware_ + '/train/',
    batch_size=batch_size_,
    temporal_length=temporal_length_,
    N_C=classes_)

train_generator = train_generator_obj.generate()

validation_generator_obj = CustomDataGenerator(
    data_path_context=context_aware_ + '/val/',
    data_path_action=action_aware_ + '/val/',
    batch_size=batch_size_,
    temporal_length=temporal_length_,
    N_C=classes_)

validation_generator = validation_generator_obj.generate()


#print("Assuming %d output classes" % train_generator.nb_class)
samples_per_epoch = samples_per_epoch_ or train_generator_obj.data_size // batch_size_
#samples_per_epoch -= (samples_per_epoch % batch_size_)
num_val_samples = num_val_samples_ or validation_generator_obj.data_size // batch_size_
#num_val_samples -= (num_val_samples % batch_size_)

print("Starting to train...")
model.fit_generator(train_generator,
                    steps_per_epoch=samples_per_epoch,
                    epochs=epochs_,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=validation_generator, 
                    validation_steps=num_val_samples,
                    workers=workers_
                    )

model.save_weights('data/model_weights/ms_lstm_final.h5')
