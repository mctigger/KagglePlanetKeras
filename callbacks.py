import json
import os
from keras.callbacks import Callback
from keras.models import save_model

import paths


class LossHistory(Callback):
    def __init__(self, name, key, save_model=False):
        self.name = name
        self.save_model = save_model
        self.key = key

        Callback.__init__(self)

    def on_epoch_end(self, epoch, logs=None):
        file_name = paths.logs + self.name + '_' + self.key + '.json'

        loss = float(logs.get(self.key))

        if os.path.isfile(paths.logs + self.name + '_' + self.key + '.json'):
            with open(file_name, 'r', encoding='utf-8') as outfile:
                losses = json.load(outfile)

        else:
            losses = []

        with open(file_name, 'w', encoding='utf-8') as outfile:
            if self.save_model and len(losses) == 0 or loss < min(losses):
                save_model(self.model, paths.models + self.name)

            losses.append(loss)
            json.dump(losses, outfile, sort_keys=True, indent=2)
