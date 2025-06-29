import tensorflow as tf

# Load from SavedModel format
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
tflite_model = converter.convert()

with open("asl_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model conversion complete.")
