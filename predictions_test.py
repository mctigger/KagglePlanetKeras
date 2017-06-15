import numpy as np
from keras.models import load_model

import paths
import os
import load_data

import infer_val

model_name = 'nn_128_dense_global_max_upsampling'
model = load_model(paths.models + model_name)

# Predict treshold based on validation data
print('Calculating thresholds')
thresholds, val_predictions = infer_val.infer_threshold(model, 40400)

predictions = np.load(paths.predictions + model_name + '.npy')

print(predictions.shape)

# Apply threshold onto predictions
predictions[predictions > thresholds] = 1
predictions[predictions <= thresholds] = 0

print(predictions.shape)

test_file_ids = list(map(lambda x: x[:-4], os.listdir(paths.test_jpg)))

print(len(test_file_ids))

print('Creating submission')
rows = ['image_name,tags']
for p, file in zip(predictions, test_file_ids):
    row = file + ','
    for tag, probability in zip(load_data.tags + load_data.tags_weather, p):
        if probability == 1:
            row += tag + ' '
    if row != file + ',':
        row = row[:-1]
    rows.append(row)

print(len(rows))

with open(paths.submissions + model_name, 'w') as submission:
    for item in rows:
        submission.write("%s\n" % item)
        print(item)

    print('Submission ready!')

exit()
