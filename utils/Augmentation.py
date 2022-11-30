# -*- encoding: utf-8 -*-
'''
@File    :   Augmentation.py
@Time    :   2022/11/30 16:49:55
@Author  :   Jim Hsu 
@Version :   1.0
@Contact :   jimhsu11@gmail.com
'''

# here put the import lib
import tensorflow as tf
from tensorflow.image import (
    random_flip_left_right,
    random_flip_up_down,
    random_brightness,
    random_contrast,
    random_saturation,
    random_hue,
    random_crop,
    random_jpeg_quality,
    rot90,
    resize,
    resize_with_crop_or_pad,
)
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomRotation,
)

class DataAug:  # 設定常用的 augment 方式
    seed = 1234
    # ColorJitter
    brightness = 0.05       # 建議數字 < 0.2
    contrast_lower = 1.0    # 建議 > 0.7
    contrast_upper = 1.0    # 建議 < 1.3
    hue = 0.0               # 建議 < 0.02
    saturation_lower = 0.6  # 建議 > 0.8
    saturation_upper = 1.6  # 建議 < 1.2
    # Crop
    zoom = 1.3
    # Zoom
    scale_minval = 0.95
    scale_maxval = 1.05
    # Rotation
    rotation_factor = 0.1
    
    translate = (0.01, 0.01)  # 0~1 之間，
    padding_size = 20


# 隨機調整圖片的亮度(brightness)、對比(contrast)、飽和度(saturation)和色調(hue)。
def ColorJitter(x):
    x = random_brightness(x, DataAug.brightness)
    x = random_contrast(x, DataAug.contrast_lower, DataAug.contrast_upper)
    x = random_hue(x, DataAug.hue)
    x = random_saturation(x, DataAug.saturation_lower, DataAug.saturation_upper)
    # x = random_brightness(x, DataAug.brightness, DataAug.seed)
    # x = random_contrast(x, DataAug.contrast_lower, DataAug.contrast_upper, DataAug.seed)
    # x = random_hue(x, DataAug.hue, DataAug.seed)
    # x = random_saturation(x, DataAug.saturation_lower, DataAug.saturation_upper, DataAug.seed)
    return x

# 隨機水平翻轉、垂直翻轉
def Flip(x):
    x = random_flip_up_down(x)
    x = random_flip_left_right(x)
    # x = random_flip_up_down(x, DataAug.seed)
    # x = random_flip_left_right(x, DataAug.seed)
    return x

# 裁剪
def Crop(x):
    # NOTE 如果放大太多可能會導致缺陷消失在圖片上
    height, width, _ = x.shape
    x = resize(x, [int(height*DataAug.zoom), int(width*DataAug.zoom)])
    x = random_crop(x, [height, width, 3])
    # x = random_crop(x, [height, width, 3], DataAug.seed)
    return x

# 隨機縮放
def Zoom(x):
    height, width, channel = x.shape
    scale = tf.random.uniform([], DataAug.scale_minval, DataAug.scale_maxval)
    new_size = (scale*height, scale*width)
    x = resize(x, new_size)
    x = resize_with_crop_or_pad(x, height, width)
    return x

# 隨機旋轉
def Rotation(x):
    RotationLayer = RandomRotation(DataAug.rotation_factor, seed=DataAug.seed)
    x = RotationLayer(x)
    return x
# k = tf.random.uniform([], 0, 4, tf.int32)
# x = tf.image.rot90(x, k)
