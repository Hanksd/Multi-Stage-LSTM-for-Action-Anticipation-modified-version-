from models import MS_LSTM
import numpy as np
import argparse
from feature_generator2 import CustomDataGenerator
from keras import backend as K
from keras.optimizers import SGD

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
    "--model",
    required=True,
    type=str,
    help="model you want to test")

parser.add_argument(
    "--classes",
    type=int,
    default=21,
    help="number of classes in target dataset")

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
model_ = 'data/model_weights/mslstm_best.h5'
classes_ = 21
temporal_length_ = 40
cell_ =  2048

loss_ = 'crossentropy'
learning_rate_ = 0.0001


def totally_linear(y_true, y_pred):
        exp_loss = 0
        T = 18
        for t in range(1,21):
                exp_loss = exp_loss + ((np.double(t)/(T)) * (K.categorical_crossentropy(y_pred, y_true)))

        return exp_loss


def totally_expontial(y_true, y_pred):
    exp_loss = 0
    T = 18
    for t in range(0, 21):
        exp_loss = exp_loss + (np.exp((-1) * (T - t)) * K.categorical_crossentropy(y_pred, y_true))

    return exp_loss


def partially_linear(true_dist, coding_dist):
        loss = 0
        TIME = 50
        N_C = 21
        batch = 32
        for t in range (TIME):
                term1 = true_dist[:,t] * K.log(coding_dist[:,t]+0.0000001)
                term2 = (1-true_dist[:,t]) * K.log(1-coding_dist[:,t]+0.0000001)
                loss = loss + np.double(1)/N_C * K.sum(term1+term2*np.double(t)/TIME, axis=1)

        return -loss/batch

def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)

    return K.maximum(0., neg - pos + 1.)


def categorical_crossentropy(y_true, y_pred):

    return K.categorical_crossentropy(y_true, y_pred)




model = MS_LSTM(INPUT_LEN=temporal_length_,
                INPUT_DIM=4096,
                OUTPUT_LEN=classes_,
                cells=cell_)


model.load_weights(model_)

sgd = SGD(lr=learning_rate_, momentum=0.9, nesterov=True)

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



validation_generator_obj = CustomDataGenerator(
    data_path_context=context_aware_ + '/val/',
    data_path_action=action_aware_ + '/val/',
    batch_size=1,
    temporal_length=temporal_length_,
    N_C=classes_)

validation_generator = validation_generator_obj.generate()

performance_w_avg = np.zeros((temporal_length_,1))
performance_wo_avg = np.zeros((temporal_length_,1))
anticipation = np.zeros((validation_generator_obj.data_size, temporal_length_, 1))

y_test = np.zeros((validation_generator_obj.data_size,temporal_length_, classes_))
x_test_context = np.zeros((validation_generator_obj.data_size,temporal_length_, 4096))
x_test_action = np.zeros((validation_generator_obj.data_size,temporal_length_, 4096))

for index, item in enumerate(validation_generator):

    x_test_context[index] = item[0]
    x_test_action[index] = item[1]
    #x_test_context = item[0]
    #x_test_action = item[1]
    #print(index)
    y_test[index] = item[-1]

gt = np.argmax(y_test, axis=2)

for t in range(1,temporal_length_):

    x_context = np.zeros((validation_generator_obj.data_size, temporal_length_, 4096))
    x_action = np.zeros((validation_generator_obj.data_size, temporal_length_, 4096))

    x_context[:, :t, :] = x_test_context[:, :t, :]
    x_action[:, :t, :] = x_test_action[:, :t, :]

    out = model.evaluate([x_context, x_action], [y_test, y_test])

    pred = model.predict([x_context, x_action])
    #print(pred.shape)
    prediction = pred[0]

    avg = np.mean(prediction[:, :t, :], axis=1)
    anticipation[:, t, 0] = np.argmax(avg, axis=1)

    correct = 0
    incorrect = 0

    for sample in range(validation_generator_obj.data_size):
        if anticipation[sample, t] == gt[sample, t]:
            correct += 1
        else:
            incorrect += 1

    performance_w_avg[t] = np.double(correct) / (correct + incorrect)

    performance_wo_avg[t] = out[-2]
    #break


for i in range(temporal_length_):

    print('w/ Temporal Average Pooling: ' + str(performance_w_avg[i][0]) +
          ' -- wo/ Temporal Average Pooling: ' + str(performance_wo_avg[i][0]))
    #break

