from sklearn.model_selection import train_test_split
import load_data


def chain_to_pipeline(chain, labels, batch_size):
    pipeline = load_data.apply_chain(chain)
    samples = load_data.parallel_sample_gen(pipeline)(labels)
    batches = load_data.batch_gen(samples, batch_size)

    return batches


def fit_model(
        model,
        name,
        chain_train,
        chain_val,
        epochs=150,
        batch_size=72,
        callbacks=None
):
    batch_size_train = batch_size
    batch_size_val = (batch_size_train*2)

    print('Loading labels...')
    labels = load_data.get_labels()

    print('Creating train/val split')
    x_train, x_test, y_train, y_test = load_data.train_test_split(labels)

    train = load_data.cycle(list(zip(x_train, y_train)), True)
    test = load_data.cycle(list(zip(x_test, y_test)), False)

    gen_train = chain_to_pipeline(chain_train, train, batch_size_train)
    gen_val = chain_to_pipeline(chain_val, test, batch_size_val)

    print('Starting network...')

    if callbacks is None:
        callbacks = [
            callbacks.LossHistory(name, 'val_loss', save_model=True),
            callbacks.LossHistory(name, 'loss', save_model=False)
        ]

    model.fit_generator(
        gen_train,
        steps_per_epoch=len(x_train) // batch_size_train,
        epochs=epochs,
        validation_data=gen_val,
        validation_steps=len(x_test) // batch_size_val,
        callbacks=callbacks,
        max_q_size=2
    )
