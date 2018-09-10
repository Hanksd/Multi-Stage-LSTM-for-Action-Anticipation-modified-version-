import numpy as np
from random import shuffle
import glob


class CustomDataGenerator(object):

    def __init__(self,
                 data_path_action,
                 data_path_context,
                 batch_size=32,
                 temporal_length=50,
                 N_C=21):

        self.batch_size = batch_size
        self.data_path_action = data_path_action
        self.data_path_context = data_path_context
        self.temporal_length = temporal_length
        self.classes = N_C

        self.features_action = glob.glob(self.data_path_action + '/feature_*.npy')
        self.features_action.sort()
        print(self.data_path_context)
        self.features_context = glob.glob(self.data_path_context + '/feature_*.npy')
        self.features_context.sort()

        self.labels = glob.glob(self.data_path_context + '/label_*.npy')
        self.labels.sort()

        self.pairs = list(zip(self.features_context, self.features_action, self.labels))
        shuffle(self.pairs)
        #print(self.pairs[10][0])
        #a = np.load(self.pairs[10][2])
        #print(a.shape)

        self.data_size = len(self.pairs)
        print(self.data_size)
        self.current = 0

    def generate(self):

        while True:

            if self.current < self.data_size - self.batch_size:

                X_c = np.zeros((self.batch_size,self.temporal_length, 4096))
                X_a = np.zeros((self.batch_size, self.temporal_length, 4096))
                y = np.zeros((self.batch_size, self.temporal_length, self.classes))

                cnt = 0
                for pair in range(self.current,self.current+self.batch_size):

                    X_c[cnt] = np.load(self.pairs[pair][0])
                    #print()
                    X_a[cnt] = np.load(self.pairs[pair][1])
                    y[cnt] = np.load(self.pairs[pair][2])

                    cnt += 1
                #X_c.transpose(0,2,1)
                #X_a.transpose(0,2,1)
                #self.current += self.batch_size

                yield ({'input_1':X_c, 'input_2':X_a}, {'stage1':y,'stage2': y})

                self.current += self.batch_size

            else:

                self.current = 0
                shuffle(self.pairs)

                X_c = np.zeros((self.batch_size, self.temporal_length, 4096))
                X_a = np.zeros((self.batch_size, self.temporal_length, 4096))
                y = np.zeros((self.batch_size, self.temporal_length, self.classes))

                cnt = 0
                for pair in range(self.current, self.current + self.batch_size):

                    X_c[cnt] = np.load(self.pairs[pair][0])
                    X_a[cnt] = np.load(self.pairs[pair][1])
                    y[cnt] = np.load(self.pairs[pair][2])

                    cnt += 1
                #X_c.transpose(0,2,1)
                #X_a.transpose(0,2,1)
                #self.current += self.batch_size

                #yield ({'input_1':X_c, 'input_2':X_a}, {'output_1':y,'output_2': y})
                #yield ({'input_1':X_c, 'input_2':X_a}, {'input_3':y,'input_4': y})
                yield ({'input_1':X_c, 'input_2':X_a}, {'stage1':y,'stage2': y})

                self.current += self.batch_size


'''
how to use:
train_generator = CustomDataGenerator(*params1).generator()
validation_generator = CustomDataGenerator(*params2).generator()

model.fit_generator(generator = training_generator,
                    validation_data = validation_generator,
                    nb_epoch = 50,
                    verbose = 1)

'''