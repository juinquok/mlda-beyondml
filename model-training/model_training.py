# from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import Model, callbacks

# Actions to be trained
actions = np.array(['giddy', 'few', 'months', 'uncomfortable', 'no', 'room', 'spin', 'allergic'])

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train=X
y_train=y

# Train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_seed=0)

# Create Tensorboard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Implement early stopping
early_stopping = EarlyStopping(monitor='categorical_accuracy', min_delta=0.001, patience=100, verbose=1, mode='auto', restore_best_weights=True)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback, early_stopping], verbose=1)

model.save('medcine.h5')