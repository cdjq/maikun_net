
import json
    #%%
import matplotlib.pyplot as plt
import os,glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
from tensorflow.keras import datasets, layers, models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',  padding='same',input_shape=(240, 240, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.5))  # 添加 dropout 层，丢弃率为 0.5
model.add(layers.Dense(404, activation='relu'))
model.add(layers.Reshape((101, 4)))  # 修改输出形状为 (None, 225, 3)
model.summary()
def visualize_keypoints(image, label):
    # 创建一个副本以免修改原始图像
    image_with_cells = np.copy(image)
    
    max_prob_cell1 = None
    max_prob_cell2 = None
    max_prob_cell3 = None
    max_prob_cell4 = None
    max_prob1 = -1
    max_prob2 = -1
    max_prob3 = -1
    max_prob4 = -1

    # 遍历所有单元格
    for i in range(10):
        for j in range(10):
            cellMinX = j * 24
            cellMaxX = (j + 1) * 24
            cellMinY = i * 24
            cellMaxY = (i + 1) * 24
            
            # 获取当前单元格的概率信息
            cell1_prob = label[10 * i + j][0]
            cell2_prob = label[10 * i + j][1]
            cell3_prob = label[10 * i + j][2]
            cell4_prob = label[10 * i + j][3]
            # 记录概率最大的两个单元格信息
            if cell1_prob > max_prob1:
                max_prob1 = cell1_prob
                max_prob_cell1 = (i, j)
            if cell2_prob > max_prob2:
                max_prob2 = cell2_prob
                max_prob_cell2 = (i, j)
            if cell3_prob > max_prob3:
                max_prob3 = cell3_prob
                max_prob_cell3 = (i, j)
            if cell4_prob > max_prob4:
                max_prob4 = cell4_prob
                max_prob_cell4 = (i, j)
    # 确定概率最大的两个点所在的单元格进行着色
    cell1_minX = max_prob_cell1[1] * 24
    cell1_maxX = (max_prob_cell1[1] + 1) * 24
    cell1_minY = max_prob_cell1[0] * 24
    cell1_maxY = (max_prob_cell1[0] + 1) * 24
    
    cell2_minX = max_prob_cell2[1] * 24
    cell2_maxX = (max_prob_cell2[1] + 1) * 24
    cell2_minY = max_prob_cell2[0] * 24
    cell2_maxY = (max_prob_cell2[0] + 1) * 24
    if label[100][2] < max_prob3:
       
       cell3_minX = max_prob_cell3[1] * 24
       cell3_maxX = (max_prob_cell3[1] + 1) * 24
       cell3_minY = max_prob_cell3[0] * 24
       cell3_maxY = (max_prob_cell3[0] + 1) * 24
       image_with_cells[cell3_minY:cell3_maxY, cell3_minX:cell3_maxX] = 56 
    if label[100][3] < max_prob4:
       cell4_minX = max_prob_cell4[1] * 24
       cell4_maxX = (max_prob_cell4[1] + 1) * 24
       cell4_minY = max_prob_cell4[0] * 24
       cell4_maxY = (max_prob_cell4[0] + 1) * 24
       # 灰色表示第二个点所在的单元格
       image_with_cells[cell4_minY:cell4_maxY, cell4_minX:cell4_maxX] = 200 
    # 白色表示第一个点所在的单元格
    image_with_cells[cell1_minY:cell1_maxY, cell1_minX:cell1_maxX] = 255  
    
    # 灰色表示第二个点所在的单元格
    image_with_cells[cell2_minY:cell2_maxY, cell2_minX:cell2_maxX] = 128 



    # 可视化图像
    plt.imshow(image_with_cells, cmap='gray')
    plt.axis('off')
    plt.show()
model.load_weights("training_1/autoCar.weights.h5")


#用训练好的模型去预测
def image_read(imname):
    # 读取图像并转换为灰度图像
    image = Image.open(imname)
    gray_image = image.convert('L')

    # 显示灰度图像
    #gray_image.show()

    # 将灰度图像转换为 numpy 数组
    gray_array = np.array(gray_image)

    # 对图像进行归一化处理
    #normalized_image = (gray_array / 255.0) * 2.0 - 1.0 

    return gray_array
def  valModel(img):
    # 读取图像并转换为灰度图像
    images=[]
    for k in range(120, 127):
        image_path = f"./data/val/photo{k}.png"
        label_path = f"./data/val/photo{k}.json"
        
        # 读取图像数据并转换为灰度图像
        gray_image = image_read(image_path)
        
        # 将图像和关键点附加到列表中
        images.append(gray_image)
    images = tf.cast(images, tf.float32)
    print(images.shape)
    result = model.predict(images)
    print(result.shape)
    return images,result
    
imageVal,label = valModel("./data/val/photo122.png")
visualize_keypoints(imageVal[0],label[0])
visualize_keypoints(imageVal[1],label[1])
visualize_keypoints(imageVal[2],label[2])
print(label[2])
visualize_keypoints(imageVal[3],label[3])
visualize_keypoints(imageVal[4],label[4])
visualize_keypoints(imageVal[5],label[5])
visualize_keypoints(imageVal[6],label[6])
print(label[6])
#print(label[6])