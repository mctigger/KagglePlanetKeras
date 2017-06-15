from pathos.multiprocessing import Pool
import numpy as np
from skimage.transform import AffineTransform, SimilarityTransform, warp
import pandas as pd
from skimage import io
from scipy.misc import imresize
import paths
from threading import Semaphore
import sklearn.model_selection

tags = [
    'blooming',
    'selective_logging',
    'blow_down',
    'conventional_mine',
    'bare_ground',
    'artisinal_mine',
    'primary',
    'agriculture',
    'water',
    'habitation',
    'road',
    'cultivation',
    'slash_burn'
]

tags_weather = [
    'cloudy',
    'partly_cloudy',
    'haze',
    'clear'
]


def cycle(items, shuffle=True):
    l = []
    for item in items:
        l.append(item)
        yield item

    print('\n Loaded all items into memory...')

    while True:
        if shuffle:
            np.random.shuffle(l)

        for item in l:
            yield item


def apply_chain(chain):
    def call(sample):
        np.random.seed()

        x, y = sample

        for fn in chain:
            x, y = fn(x, y)

        return x, y

    return call


def labels_to_xy(labels):
    samples = list(samples_non_load(labels))
    x = list(map(lambda sample: sample[0], samples))
    y = list(map(lambda sample: sample[1], samples))

    return x, y



def load_jpg(path):
    return io.imread(path) / 255


center_shift = 256 / 2
tf_center = SimilarityTransform(translation=-center_shift)
tf_uncenter = SimilarityTransform(translation=center_shift)


def load():
    def call(x, y):
        x = load_jpg(x)

        return x, y

    return call


def augment(
        rotation_fn=lambda: np.random.random_integers(0, 360),
        translation_fn=lambda: (np.random.random_integers(-10, 10), np.random.random_integers(-10, 10)),
        scale_factor_fn=lambda: np.random.random_sample() * 0.3 + 0.9,
        shear_fn=lambda: np.random.random_integers(-10, 10),
        flip_fn=lambda: np.random.rand() > .5
):
    def call(x, y):
        rotation = rotation_fn()
        translation = translation_fn()
        scale_factor = scale_factor_fn()
        scale = scale_factor, scale_factor
        shear = shear_fn()
        flip = flip_fn()

        if flip:
            shear += 180
            rotation += 180

        tf_augment = AffineTransform(scale=scale, rotation=np.deg2rad(rotation), translation=translation, shear=np.deg2rad(shear))
        tf = tf_center + tf_augment + tf_uncenter

        x = warp(x, tf, order=1, preserve_range=True, mode='symmetric')

        return x, y

    return call


def resize(output_size):
    def call(x, y):
        x = x[::256//output_size, ::256//output_size, :]
        return x, y

    return call


def _parallel_helper(samples, semaphore):
    for sample in samples:
        semaphore.acquire()
        yield sample


def parallel_sample_gen(fn, queue_size=100, workers=12):
    def call(samples):
        p = Pool(workers)
        semaphore = Semaphore(queue_size)
        samples = _parallel_helper(samples, semaphore)

        for sample in p.imap(fn, samples):
            semaphore.release()
            yield sample

        p.close()
        p.join()

    return call


def get_labels():
    labels_df = pd.read_csv(paths.train_csv)
    labels_df.head()

    # Build list with unique labels
    label_list = []
    for tag_str in labels_df.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)

    # Add onehot features for every label
    for label in label_list:
        labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
    # Display head

    return labels_df


def train_test_split(labels, test_size=0.2):
    x, y = labels_to_xy(labels)

    return sklearn.model_selection.train_test_split(x, y, test_size=test_size, random_state=32)


def samples_non_load(labels):
    for i, series in labels.iterrows():
        try:
            x = paths.train_jpg + series.get('image_name') + '.jpg'
            y = [series.get(tag) for tag in tags+tags_weather]
            yield x, y

        except ValueError:
            print('\nInvalid file:', paths.train_jpg + series.get('image_name') + '.jpg')


def batch_gen(sample_gen, batch_size):
    x_batch = []
    y_batch = []

    for x, y in sample_gen:
        x_batch.append(x)
        y_batch.append(y)

        if len(x_batch) >= batch_size:
            yield np.stack(x_batch), np.stack(y_batch)
            x_batch = []
            y_batch = []


def batch_gen_x(sample_gen, batch_size):
    x_batch = []

    for x in sample_gen:
        x_batch.append(x)

        if len(x_batch) >= batch_size:
            yield np.stack(x_batch)
            x_batch = []

    else:
        yield np.stack(x_batch)
