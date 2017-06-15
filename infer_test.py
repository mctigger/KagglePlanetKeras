import os

import numpy as np
from itertools import cycle

import paths
import load_data


def predict(model, rotation_times=0):
    test_file_paths = list(os.listdir(paths.test_jpg))

    print('len(test_file_paths)', len(test_file_paths))

    def get_images(test_file_paths):
        for i, path in cycle(enumerate(test_file_paths)):
            test_file = load_data.load_jpg(paths.test_jpg + path)
            test_file = test_file[::2, ::2, :]
            print('Test images predicted: ' + str(i), end='\r')

            test_file = np.rot90(test_file, rotation_times)

            yield test_file

    predictions = model.predict_generator(load_data.batch_gen_x(get_images(test_file_paths), 64), steps=61191 // 64 + 63) #61191

    prediction_collector = []
    for prediction, file_name in zip(predictions, test_file_paths):
        prediction_collector.append(prediction)
        print(len(prediction_collector))

    print(len(prediction_collector))

    return np.array(prediction_collector)
