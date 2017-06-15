from sklearn.model_selection import train_test_split

import load_data

labels = load_data.get_labels()
samples = list(load_data.samples_non_load(labels))

X = list(map(lambda sample: sample[0], samples))
y = list(map(lambda sample: sample[1], samples))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

train = list(zip(X_train, y_train))

print(y_train)
print(y_test)
