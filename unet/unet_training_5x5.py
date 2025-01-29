import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Disable GPU
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
)

from keras.models import Model
from keras.optimizers import Adam, schedules
from matplotlib import pyplot as plt

# Check for GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class Unet:
    """UNet model for image segmentation."""

    def __init__(
        self,
        input_shape: tuple,
        filters: int = 16,
        dropout: float = 0.05,
        batch_normalisation: bool = True,
        trained_model: str = None,
        image: np.ndarray = None,
        layers: int = 4,
        output_activation: str = "sigmoid",
        model: Model = None,
        reconstructed: np.ndarray = None,
    ):
        """
        Initialise the UNet model.

        Parameters
        ----------
        input_shape : tuple
            The shape of the input image.
        filters : int
            The number of filters to use in the convolutional layers, default is 16.
        dropout : float
            The dropout rate, default is 0.05.
        batch_normalisation : bool
            Whether to use batch normalisation, default is True.
        trained_model : str
            The path to a trained model.
        image : np.ndarray
            The image to decode. Image must be 2D given as 4D numpy array, e.g. (1, 256, 256, 1).
            Image must be grayscale, e.g. not (1, 256, 256, 3). Image array row columns must
            be divisible by 2^layers, e.g. 256 % 2^4 == 0.
        layers : int
            The number of encoding and decoding layers, default is 4.
        output_activation : str
            The activation function for the output layer, either sigmoid or softmax.
            Default is sigmoid.
        model : keras.models.Model
            A pre-built model, populated by the build_model method.
        reconstructed : np.ndarray
            The reconstructed image, created by the decode_image method.
        """
        self.input_shape = input_shape
        self.filters = filters
        self.dropout = dropout
        self.batch_normalisation = batch_normalisation
        self.trained_model = trained_model
        self.image = image
        self.layers = layers
        self.output_activation = output_activation
        self.model = model
        self.reconstructed = reconstructed

        self.model = self.build_model()

    def convolutional_block(self, input_tensor, filters, kernel_size=5):
        """Convolutional block for UNet."""
        convolutional_layer = Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
        )
        batch_normalisation_layer = BatchNormalization()
        relu_layer = Activation("relu")

        if self.batch_normalisation:
            return relu_layer(batch_normalisation_layer(convolutional_layer(input_tensor)))
        return relu_layer(convolutional_layer(input_tensor))

    def encoding_block(self, input_tensor, filters, kernel_size=3):
        """Encoding block for UNet."""
        convolutional_block = self.convolutional_block(input_tensor, filters, kernel_size)
        max_pooling_layer = MaxPooling2D((2, 2), padding="same")
        dropout_layer = Dropout(self.dropout)

        return convolutional_block, dropout_layer(max_pooling_layer(convolutional_block))

    def decoding_block(self, input_tensor, concat_tensor, filters, kernel_size=5):
        """Decoding block for UNet."""
        transpose_convolutional_layer = Conv2DTranspose(
            filters, (kernel_size, kernel_size), strides=(2, 2), padding="same"
        )
        skip_connection = Concatenate()(
            [transpose_convolutional_layer(input_tensor), concat_tensor]
        )
        dropout_layer = Dropout(self.dropout)
        return self.convolutional_block(dropout_layer(skip_connection), filters, kernel_size)

    def build_model(self):
        """Build the UNet model."""
        input_image = Input(self.input_shape, name="img")
        current = input_image

        # Encoding Path
        convolutional_tensors = []
        for layer in range(self.layers):
            convolutional_tensor, current = self.encoding_block(
                current, self.filters * (2 ** layer)
            )
            convolutional_tensors.append((convolutional_tensor))

        # Latent Convolutional Block
        latent_convolutional_tensor = self.convolutional_block(
            current, filters=self.filters * 2 ** self.layers
        )

        # Decoding Path
        current = latent_convolutional_tensor
        for layer in reversed(range(self.layers)):
            current = self.decoding_block(
                current, convolutional_tensors[layer], self.filters * (2 ** layer)
            )

        outputs = Conv2D(1, (1, 1), activation=self.output_activation)(current)
        model = Model(inputs=[input_image], outputs=[outputs])
        return model

    def compile_model(self):
        """Compile the UNet model."""
        self.model.compile(
            optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", "iou_score"]
        )
        return self.model

    def decode_image(self):
        """Returns images decoded by a trained model."""
        print(f"Predicting source segmentation using pre-trained model...")
        if self.trained_model is None or self.image is None:
            raise ValueError("Trained model and image arguments are required to decode image.")
        if isinstance(self.image, np.ndarray) is False:
            raise TypeError("Image must be a numpy array.")
        if len(self.image.shape) != 4:
            raise ValueError("Image must be 4D numpy array for example (1, 256, 256, 1).")
        if self.image.shape[3] != 1:
            raise ValueError("Input image must be grayscale.")
        if (
            self.image.shape[0] % 2 ** self.layers != 0
            and self.image.shape[1] % 2 ** self.layers != 0
        ):
            raise ValueError("Image shape should be divisible by 2^layers.")

        self.model = self.compile_model()
        self.model.load_weights(self.trained_model)
        self.reconstructed = self.model.predict(self.image)
        return self.reconstructed

train_file = "unet/data/training_data.npy"
validation_file = "unet/data/validation_data.npy"
test_file = "unet/data/test_data.npy"

# def normalise_data(data):
#     """Normalise data using input data for min and max."""
#     # data_max = np.max(data[0])
#     data_max = 1e3
#     input_normalised = data[0] / data_max
#     clean_normalised = data[1] / data_max
#     contaminant_normalised = data[2] / data_max
#     return np.asarray([input_normalised, clean_normalised, contaminant_normalised])

# def load_data(file):
#     data = np.load(file)
#     data_transposed = data.transpose (1, 0, 2, 3)
#     data_normalised = np.asarray([normalise_data(data) for data in data_transposed])
#     normalised_transposed = data_normalised.transpose (1, 0, 2, 3)
#     simulations = normalised_transposed[0]
#     clean_simulations = normalised_transposed[1]
#     contaminant_simulations = normalised_transposed[2]

#     simulations_data = np.asarray(simulations, dtype=np.float32)
#     clean_simulations_data = np.asarray(clean_simulations, dtype=np.float32)
#     contaminant_simulations_data = np.asarray(contaminant_simulations, dtype=np.float32)

#     return simulations_data, clean_simulations_data, contaminant_simulations_data

# def load_data(file):
#     data = np.load(file)

#     simulations = np.asarray(data[0], dtype=np.float32)
#     clean_simulations = np.asarray(data[1], dtype=np.float32)
#     contaminant_simulations = np.asarray(data[2], dtype=np.float32)

#     return simulations, clean_simulations, contaminant_simulations

def normalise_data(data, normalise_max=300):
    """Normalise data using input data for min and max."""

    data_max = np.max(data[0])
    normalisation_factor = normalise_max / data_max
    input_normalised = data[0] * normalisation_factor
    clean_normalised = data[1] * normalisation_factor
    contaminant_normalised = data[2] * normalisation_factor
    return np.asarray([input_normalised, clean_normalised, contaminant_normalised])

def load_data(file):
    data = np.load(file)
    data_transposed = data.transpose (1, 0, 2, 3)
    data_normalised = np.asarray([normalise_data(data) for data in data_transposed])
    normalised_transposed = data_normalised.transpose (1, 0, 2, 3)
    simulations = normalised_transposed[0]
    clean_simulations = normalised_transposed[1]
    contaminant_simulations = normalised_transposed[2]

    simulations_data = np.log1p(np.asarray(simulations, dtype=np.float32))
    clean_simulations_data = np.log1p(np.asarray(clean_simulations, dtype=np.float32))
    contaminant_simulations_data = np.log1p(np.asarray(contaminant_simulations, dtype=np.float32))
    
    return simulations_data, clean_simulations_data, contaminant_simulations_data

train_simulations, train_clean_simulations, train_contaminant_simulations = load_data(train_file)
validation_simulations, validation_clean_simulations, validation_contaminant_simulations = load_data(validation_file)
test_simulations, test_clean_simulations, test_contaminant_simulations = load_data(test_file)

print(f"Train simulations: {train_simulations.shape}")
print(f"Train clean simulations: {train_clean_simulations.shape}")
print(f"Train contaminant simulations: {train_contaminant_simulations.shape}")

print(f"Validation simulations: {validation_simulations.shape}")
print(f"Validation clean simulations: {validation_clean_simulations.shape}")
print(f"Validation contaminant simulations: {validation_contaminant_simulations.shape}")

print(f"Test simulations: {test_simulations.shape}")
print(f"Test clean simulations: {test_clean_simulations.shape}")
print(f"Test contaminant simulations: {test_contaminant_simulations.shape}")


data_train = np.reshape(train_simulations, (train_simulations.shape[0], train_simulations.shape[1], train_simulations.shape[2], 1))
labels_train = np.reshape(train_contaminant_simulations, (train_contaminant_simulations.shape[0], train_contaminant_simulations.shape[1], train_contaminant_simulations.shape[2], 1))

data_validation = np.reshape(validation_simulations, (validation_simulations.shape[0], validation_simulations.shape[1], validation_simulations.shape[2], 1))
labels_validation = np.reshape(validation_contaminant_simulations, (validation_contaminant_simulations.shape[0], validation_contaminant_simulations.shape[1], validation_contaminant_simulations.shape[2], 1))

data_test = np.reshape(test_simulations, (test_simulations.shape[0], test_simulations.shape[1], test_simulations.shape[2], 1))
labels_test = np.reshape(test_contaminant_simulations, (test_contaminant_simulations.shape[0], test_contaminant_simulations.shape[1], test_contaminant_simulations.shape[2], 1))

unet_model = Unet(input_shape=(256, 2048, 1), output_activation=None, layers=6)
model = unet_model.model

decay_steps = 1000
initial_learning_rate = 0.1
warmup_steps = 1000
target_learning_rate = 3
lr_warmup_decayed_fn = schedules.CosineDecay(
    initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
    warmup_steps=warmup_steps
)

def weighted_mse(y_true, y_pred):
    mask = y_true == 0
    y_0_predicted = tf.where(mask, y_pred, tf.zeros_like(y_pred))
    y_0_true = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_1_predicted = tf.where(mask, tf.zeros_like(y_pred), y_pred)
    y_1_true = tf.where(mask, tf.zeros_like(y_true), y_true)
    return tf.reduce_mean(tf.square(y_0_predicted - y_0_true) + 10000 * tf.square(y_1_predicted - y_1_true))

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="mean_squared_error",
    metrics=["accuracy"],
)

print(model.summary())

class CustomCallBack(Callback):

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])

        # val_data = self.validation_data
        # if val_data is None:
        #     print("No validation data provided.")
        #     return

        # # Extract the images and labels from the validation data
        # x_val, y_val = val_data[:2]

        # # Get model predictions (decoded images)
        # y_pred = self.model.predict(x_val, batch_size=32)

        # # Calculate the mean of the max pixel values for predictions and labels
        # mean_max_pixel_pred = np.mean(np.max(y_pred, axis=(1, 2)))  # max per image, then mean over batch
        # mean_max_pixel_true = np.mean(np.max(y_val, axis=(1, 2)))   # max per image, then mean over batch

        # # Calculate the ratio
        # ratio = mean_max_pixel_pred / mean_max_pixel_true

        self.logs_losses = np.array({"loss": self.losses, "val_loss": self.val_losses})
        
        np.save(f"{os.environ['HOME']}/Code/ExoTiC-NEAT-training/unet/training_loss_5x5.npy", self.logs_losses)

        fig, ax = plt.subplots(1, figsize=(5, 4))
        ax.plot(self.losses, label="training loss", color="#57d4c1", marker="None")
        ax.plot(self.val_losses, label="validation loss", color="#8d57d4", marker="None")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()

        plt.savefig(f"{os.environ['HOME']}/Code/ExoTiC-NEAT-training/unet/training_loss_plot_5x5.png")

        ax.set_yscale("log")
        plt.savefig(f"{os.environ['HOME']}/Code/ExoTiC-NEAT-training/unet/training_loss_plot_5x5_log.png")

        plt.close()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(
        "unet/trained_model_5x5.h5",
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    ),
    CustomCallBack(),
]

results = model.fit(
    data_train,
    labels_train,
    batch_size=8,
    epochs=200,
    callbacks=callbacks,
    validation_data=(data_validation, labels_validation),
)


best_model_epoch, best_model_val_loss = np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"])

fig, ax = plt.subplots(1, figsize=(5, 4))

ax.plot(results.history["loss"], label="training loss", color="#57d4c1", marker="None")
ax.plot(results.history["val_loss"], label="validation loss", color="#8d57d4", marker="None")
ax.plot(best_model_epoch, best_model_val_loss, marker="x", label="best model", color="#d457bf", linestyle="None", markersize=5)

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_yscale("log")
ax.legend()

plt.savefig("unet/loss_plot_5x5.png")

# decoded_images = model.predict(data_test, verbose=1)
# np.save("unet/decoded_images.npy", decoded_images)
