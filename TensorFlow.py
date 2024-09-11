import tensorflow as tf
from keras import layers, models
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from time import time

# Load Fashion-MNIST data directly from TensorFlow
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to a 1D array
    layers.Dense(128, activation='relu'),   # Dense layer with 128 units and ReLU activation
    layers.Dense(10, activation='softmax', name="LastLayer")  # Output layer with 10 units (classes) and softmax activation
])

# Normalize pixel values to be between 0 and 1
x_train, x_test = train_images / 255.0, test_images / 255.0

# Compile the model
with tf.device('/GPU:0'):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_train = tf.constant(tf.cast(x_train, dtype=tf.float32))
    y_train = tf.constant(tf.cast(train_labels, dtype=tf.float32))
    x_test = tf.constant(tf.cast(x_test, dtype=tf.float32))
    y_test = tf.constant(tf.cast(test_labels, dtype=tf.float32))

# Train the model
model.fit(x_train, y_train, epochs=5)

# Save model in the saved_model format
SAVED_MODEL_DIR = "./models/native_saved_model"
tf.saved_model.save(model, SAVED_MODEL_DIR)

# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=SAVED_MODEL_DIR,
    precision_mode=trt.TrtPrecisionMode.FP32
)

# Convert the model into TRT compatible segments
trt_func = converter.convert()
converter.summary()

MAX_BATCH_SIZE = 100
def input_fn():
    batch_size = MAX_BATCH_SIZE
    x = x_test[0:batch_size, :]
    yield [x]

converter.build(input_fn=input_fn)

OUTPUT_SAVED_MODEL_DIR = "./models/tftrt_saved_model"
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)

with tf.device('/GPU:0'):
    optimized_model = tf.saved_model.load(OUTPUT_SAVED_MODEL_DIR)

# print("The signature keys are: ",list(optimized_model.signatures.keys()))
infer = optimized_model.signatures["serving_default"]

# Evaluate your model accuracy
with tf.device('/GPU:0'):
    start = time()
    score = model.evaluate(x_test, y_test)
    u_time = time() - start

    output = []
    start = time()
    # Get batches of test data and run inference through them
    infer_batch_size = MAX_BATCH_SIZE
    for i in range(100):
        print(f"Step: {i}")

        start_idx = i * infer_batch_size
        end_idx = (i + 1) * infer_batch_size
        x_batch = x_test[start_idx:end_idx, :]

        result = infer(x_batch)
        predictions = tf.argmax(result['LastLayer'], axis=1).numpy()
        output.extend(predictions)

    o_time = time() - start

count = 0
for i in range(len(output)):
    if test_labels[i] == output[i]:
        count += 1

print('\n'*2)
print("Unoptimized Accuracy: ", 100* score[1])
print("Unoptimized Model Run Time: ", u_time)
print("Optimized Accuracy: ", 100 * count / len(test_labels))
print("Optimized Model Run Time: ", o_time)