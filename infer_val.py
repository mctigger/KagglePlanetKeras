import numpy as np
import load_data
import find_best_threshold


def chain_to_pipeline(chain, labels):
    pipeline = load_data.apply_chain(chain)
    samples = load_data.parallel_sample_gen(pipeline)(labels)

    return samples


def infer_threshold(model, labels, rotations=[0, 90, 180, 270]):
    x_test_labels, y_test = load_data.labels_to_xy(labels)

    predictions = list(map(lambda x: get_predictions(model, x_test_labels, y_test, x), rotations))

    print(len(predictions))

    predictions = np.stack(predictions)
    predictions = np.mean(predictions, axis=0)

    y = np.array(y_test)

    print(predictions.shape)
    print(y.shape)

    threshold = find_best_threshold.optimise_f2_thresholds(y, predictions)

    return threshold, predictions


def get_predictions(model, x_test_labels, y_test, rotation):
    batch_size = 128

    # Reset augmentations except for rotation
    chain = [load_data.load(), load_data.augment(
        rotation_fn=lambda: rotation,
        translation_fn=lambda: 0,
        scale_factor_fn=lambda: 1,
        shear_fn=lambda: 0,
        flip_fn=lambda: False
    ), load_data.resize(128)]

    labels = load_data.cycle(list(zip(x_test_labels, y_test)), False)
    data = chain_to_pipeline(chain, labels)
    x = map(lambda sample: sample[0], data)

    # Generate predictions. Because of batching this will result in too many predictions
    predictions = model.predict_generator(
        generator=load_data.batch_gen_x(x, batch_size),
        steps=len(x_test_labels) // batch_size + 1
    )

    # Slice prediction rest
    predictions = predictions[:len(x_test_labels)]

    return predictions

