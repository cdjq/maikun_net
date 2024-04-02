import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape
from tensorflow.keras import models
import onnx

# 构建模型
def build_model():
    inputs = tf.keras.Input(shape=(240, 240, 1))
    x = Conv2D(128, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(288, activation='relu')(x)
    outputs = Reshape((144, 2))(x)
    model = tf.keras.Model(inputs, outputs)
    return model

# 加载模型权重
model = build_model()
model.load_weights("training_1/autoCar.weights.h5")


# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
   f.write(tflite_model)
'''

# 转换模型为 ONNX 格式
import tf2onnx
# 定义输入签名
input_signature = (tf.TensorSpec((1, 240,240, 1), tf.float32),)
# 设置 Opset 版本为 19
target_opset = 15
# Convert the Keras model to ONNX
# 转换模型为 ONNX 格式，指定 Opset 版本为 19
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=target_opset)

# 保存 ONNX 模型
onnx.save_model(onnx_model, "autoCar.onnx")
'''