import load_data
import time
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import imshow


def get_samples_augmented(
        batch_size=72,
        augmentation_fn=None
):
    def call(labels):
        samples = load_data.parallel_sample_gen(augmentation_fn)(labels)
        batches = load_data.batch_gen(samples, batch_size)

        return batches

    return call


labels = load_data.get_labels()[2:8]
samples = list(load_data.samples_non_load(labels))

X = list(map(lambda sample: sample[0], samples))
y = list(map(lambda sample: sample[1], samples))

print('Creating train/val split')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

train = load_data.cycle(list(zip(X_train, y_train)), True)
test = load_data.cycle(list(zip(X_test, y_test)), False)


chain_train = [load_data.load(), load_data.augment_color_fn(), load_data.augment(), load_data.resize(256)]
sample_fn_train = load_data.apply_chain(chain_train)

chain_val = [load_data.load(), load_data.resize(256)]
sample_fn_val = load_data.apply_chain(chain_val)

start = time.time()
for i, batch in enumerate(get_samples_augmented(batch_size=1, augmentation_fn=sample_fn_train)(train)):
    x, y = batch
    #print(i*32 / (time.time() - start))
    for img in x:
        print(np.mean(img))
        print(np.max(img))
        imshow(img)
