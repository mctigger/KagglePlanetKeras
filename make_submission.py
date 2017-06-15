import numpy as np
from keras.models import load_model

import paths
import os
import sys
import load_data

import infer_val
import infer_test

args = sys.argv
model_name = args[1]

model = load_model(paths.models + model_name)

# Predict treshold based on validation data
print('Calculating thresholds')
# Calculating thresolds based on the whole data is bad depending on the model, change that!
labels = load_data.get_labels()
thresholds, val_predictions = infer_val.infer_threshold(model, labels, [0])

# Predict test data
print('Predicting test labels')
predictions = list(map(lambda r: infer_test.predict(model, r), [0]))
predictions = np.mean(np.stack(predictions), axis=0)

# Apply threshold onto predictions
predictions[predictions > thresholds] = 1
predictions[predictions <= thresholds] = 0

test_file_ids = list(map(lambda x: x[:-4], os.listdir(paths.test_jpg)))

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

with open(paths.submissions + model_name, 'w') as submission:
    for item in rows:
        submission.write("%s\n" % item)

    print('Submission ready!')
