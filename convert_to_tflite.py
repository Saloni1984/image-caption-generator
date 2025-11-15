import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model("mymodel.h5")

# Create a TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Allow TensorFlow ops that TFLite cannot natively convert
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # default TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # allow TF ops
]

# Disable lowering tensor list ops (needed for dynamic RNNs)
converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_model = converter.convert()

# Save the converted model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as model.tflite")
