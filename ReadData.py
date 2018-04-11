from __future__ import print_function
import numpy as np
import tensorflow as tf
import math as math
import os

class ReadData:

    #filenames = ["C:\\Users\\ekriesw\\Documents\\visual studio 2017\\Projects\\FirstModel\\FirstModel\\Data\\Training\\data.csv"]
    trainingfileNames= ['data.csv']
    testingfileNames = []
    buildFilePath = True
    summaries_dir = os.path.dirname(os.path.realpath(__file__)) + "\\logs"

    def filePathConstructor(self):
        # transform relative path into full path
        if (self.buildFilePath == False):
            return trainingfileNames,testingfileNames
        else :
            dir_path = os.path.dirname(os.path.realpath(__file__))
            trainingFolderPath =os.path.join(dir_path,'Data','Training')
            testingFolderPath = os.path.join(dir_path,'Data','Test')
            trainingFiles = []
            testingFiles = []
            for file in self.trainingfileNames:
                trainingFiles.append(os.path.join(trainingFolderPath, file))
            for file in self.testingfileNames:
                trainingFiles.append(os.path.join(testingFolderPath, file)) 
            return trainingFiles,testingFiles

    def read_my_file_format(self,filename_queue):
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        # Default values, in case of empty columns. Also specifies the type of the
        # decoded result.
        record_defaults = [[""], [""]]
        name, gender = tf.decode_csv(
            value, record_defaults=record_defaults,field_delim=',')
        features = tf.stack([name],axis=0)
        return features

    def input_pipeline(self,filenames, batch_size, num_epochs=None):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
        example = self.read_my_file_format(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        example_batch = tf.train.shuffle_batch(
            [example], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return example_batch

    