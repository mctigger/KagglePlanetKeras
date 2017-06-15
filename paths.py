import os

test_jpg = './test-jpg/'
train_jpg = './train-jpg/'

logs = './logs/'
models = './models/'
submissions = './submissions/'

train_csv = './train.csv'

dirs = [logs, models, submissions]
data = [test_jpg, train_jpg]
files = [train_csv]

for supplementary_dir in dirs:
    if os.path.isdir(supplementary_dir):
        continue

    if not os.path.isfile(supplementary_dir[:-1]):
        os.makedirs(supplementary_dir)
        print('Created directory', supplementary_dir)

    else:
        print('Path {} already exists and is not a directory. Please delete this file or change the path.'.format(supplementory_dir))


for data_dir in data:
    if os.path.isdir(data_dir):
        continue

    else:
        print('Directoy {} does not exists. Please either put the training/test data in the appropriate directories or '
              'change the path.'.format(data_dir))


for file in files:
    if os.path.isfile(file):
        continue

    else:
        print('File {} does not exists. Please either put the file in the appropriate directories or '
              'change the path.'.format(file))
