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

    
                
def parse_annotation():
    images, keypoints = [], []
    labels = []
    for k in range(0, 140):
        image_path = f"./data/train/photo{k}.png"
        label_path = f"./data/train/photo{k}.json"
        
        # 读取图像数据并转换为灰度图像
        gray_image = image_read(image_path)
        
        with open(label_path, 'r') as f:
            json_data = json.load(f)
        
        # 从JSON数据中提取关键点
        keypoint_up, keypoint_down, keypoint_T, keypoint_Cir = None, None,None,None
        for shape in json_data['shapes']:
            if shape['label'] == 'up':
                keypoint_up = shape['points'][0]
            elif shape['label'] == 'down':
                keypoint_down = shape['points'][0]
            elif shape['label'] == 'T':
                keypoint_T = shape['points'][0]
            elif shape['label'] == 'Cir':
                keypoint_Cir = shape['points'][0]
        images.append(gray_image)
        #print(keypoint_up)
        #print(keypoint_down)
        x1, y1 = keypoint_up
        x2, y2 = keypoint_down
        x3, y3 = 0,0
        x4, y4 = 0,0
        if keypoint_T ==   None:
          x3, y3 = 255,255
        else:
          x3, y3 = keypoint_T
        if keypoint_Cir ==   None:
          x4, y4 = 255,255
        else:
          x4, y4 = keypoint_Cir
          
        label = []
        for i in range(0, 10):
            for j in range(0, 10):
                cellMinX = j * 24
                cellMaxX = j * 24 + 24
                cellMinY = i * 24
                cellMaxY = i * 24 + 24
                cell = [0.0, 0.0, 0.0,0.0]  # 初始化cell数组，默认情况下，所有关键点都不在单元格中
                # 检查关键点 1 是否位于单元格中
                if cellMinX <= x1 < cellMaxX and cellMinY <= y1 < cellMaxY:
                    #print(i)
                    cell[0] = 1.0
                # 检查关键点 2 是否位于单元格中
                if cellMinX <= x2 < cellMaxX and cellMinY <= y2 < cellMaxY:
                    cell[1] = 1.0
                if cellMinX <= x3 < cellMaxX and cellMinY <= y3 < cellMaxY:
                    cell[2] = 1.0
                if cellMinX <= x4 < cellMaxX and cellMinY <= y4 < cellMaxY:
                    cell[3] = 1.0
                label.append(cell)
        cell1 = [0.0, 0.0, 0.0,0.0]
        if keypoint_T ==  None:
          cell1[2] = 1.0

        if keypoint_Cir ==  None:
          cell1[3] = 1.0
        label.append(cell1)
        labels.append(label)
    
    return images, labels


#print(keypoints)
image, label = parse_annotation()
#print(label)

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
    isprob3 =1
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
       image_with_cells[cell3_minY:cell3_maxY, cell3_minX:cell3_maxX] = 65 
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
#visualize_keypoints(image[0], label[0])
#print_array_shape(my_dict)
#visualize_keypoints(images, keypoints)

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错

print(tf.__version__)
x_train = image
y_train = label
x_test = image[-20:]
y_test = label[-20:]


np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)


x_train = tf.cast(x_train, tf.float32)/255.0
y_train = tf.cast(y_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)/255.0
y_test = tf.cast(y_test, tf.float32)

#y_train = tf.reshape(y_train,(99,675))
#y_test = tf.reshape(y_test,(20,675))
#print(x_train.shape)
print(y_test.shape)
#def parse_annotation(img_dir,ann_dir,labels):
# set random seed
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


checkpoint_path = "training_1/autoCar.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)




# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)



#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,loss="mse", metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=48, batch_size=5,
                    validation_data=(x_test, y_test),callbacks=[cp_callback])




print(tf.__version__)

#用训练好的模型去预测

def  valModel(img):
    # 读取图像并转换为灰度图像
    images=[]
    for k in range(127, 134):
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