import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Paths to datasets (update with actual paths)
TRAIN_DIR = "path_to_training_dataset"
VALIDATION_DIR = "path_to_validation_dataset"
TEST_DIR = "path_to_test_dataset"

# Image dimensions and batch size
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# Loading datasets
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

validation_generator = val_test_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Load InceptionV3 base model (pre-trained on ImageNet)
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Adding custom layers on top of the base model
model = Sequential([
    base_model,
    Flatten(),
    Dense(1024, activation="relu"),
    Dropout(0.3),
    Dense(len(train_generator.class_indices), activation="softmax")
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks for saving the best model
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5", monitor="val_loss", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    epochs=60,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save the trained model
model.save("inception_v3_sentiment_analysis.h5")
