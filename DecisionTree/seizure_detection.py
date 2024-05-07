from tensorflow.keras import layers, models

def create_seizure_detection_model(input_shape):
    """
    Create a CNN model for seizure detection.
    """
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class SeizureDetector:
    def __init__(self, model):
        self.model = model

    def train(self, eeg_data, labels, epochs=10, batch_size=32):
        """
        Train the seizure detection model.
        """
        self.model.fit(eeg_data, labels, epochs=epochs, batch_size=batch_size)

    def detect(self, eeg_epoch):
        """
        Detect if an EEG epoch contains a seizure.
        """
        return self.model.predict(eeg_epoch) > 0.5
