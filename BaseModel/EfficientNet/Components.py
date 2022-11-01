import math
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    ZeroPadding2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
)
from .utils import calculate_output_image_size

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1.0 / 3.0,
        "mode": "fan_out",
        "distribution": "uniform",
    },
}

class Conv2dStaticSamePadding(Layer):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """
    def __init__(self, filters, kernel_size, stride, groups, use_bias, image_size):
        super().__init__()
        conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding='VALID',#'SAME' if stride == 1 else 'VALID',
            groups=groups,
            use_bias=True if use_bias else False,
            kernel_initializer=CONV_KERNEL_INITIALIZER
        )
        self.conv = conv
        self.static_padding = self.tf_pad(kernel_size, stride, image_size=image_size)

    def tf_pad(self, kernel_size, stride, dilation=(1,1), image_size=None):
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = kernel_size
        sh, sw = stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * stride[0] + (kh - 1) * dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * stride[1] + (kw - 1) * dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            # TF :  ((top_pad, bottom_pad), (left_pad, right_pad))
            # torch:(left_pad, right_pad, top_pad, bottom_pad)
            static_padding = ZeroPadding2D(((pad_h // 2, pad_h - pad_h // 2), 
                                            (pad_w // 2, pad_w - pad_w // 2)))
        else:
            static_padding = tf.identity
        return static_padding

    def call(self, inputs, **kwargs):
        x = self.static_padding(inputs)
        x = self.conv(x)
        return x

class StemConv(Layer):
    def __init__(self, filters, kernel_size, stride, groups, use_bias, image_size):
        super().__init__()
        self._conv_stem = Conv2dStaticSamePadding(
            filters, 
            kernel_size, 
            stride, 
            groups, 
            use_bias, 
            image_size
        )
        self._bn0 = BatchNormalization()
        self._swish = tf.keras.activations.swish

    def call(self, inputs):
        x = self._conv_stem(inputs)
        x = self._bn0(x)
        x = self._swish(x)
        return x

class MBConvBlock(Layer):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        layer_dict (dict): BlockArgs, defined in utils.py.
        layer_weight (dict): BlockWeight, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """
    def __init__(self, filters_in, filters_out, kernel_size, strides, 
                 expand_ratio, id_skip, has_se, drop_connect_rate, image_size):
        super().__init__()
        
        self.input_filters = filters_in
        self.output_filters = filters_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.id_skip = id_skip
        self.has_se = has_se
        self.dropout_rate = drop_connect_rate
        
        # Expansion phase (Inverted Bottleneck)
        filters = filters_in * expand_ratio
        # Conv2dStaticSamePadding: filters, kernel_size, stride, groups, use_bias, image_size
        if self.expand_ratio != 1:
            self._expand_conv = Conv2dStaticSamePadding(filters, (1, 1), (1, 1), 1, False, image_size)
            self._bn0 = BatchNormalization()
        # Depthwise convolution phase
        self._depthwise_conv = Conv2dStaticSamePadding(filters, kernel_size, self.strides, filters, False, image_size)
        self._bn1 = BatchNormalization()
        image_size = calculate_output_image_size(image_size, self.strides)

        # Squeeze and Excitation layer, if desired
        filters_se = max(1, int(filters_in // 4))
        if self.has_se:
            self._se_reduce = Conv2dStaticSamePadding(filters_se, (1, 1), (1, 1), 1, True, image_size)
            self._se_expand = Conv2dStaticSamePadding(filters, (1, 1), (1, 1), 1, True, image_size)

        # Pointwise convolution phase
        self._project_conv = Conv2dStaticSamePadding(self.output_filters, (1, 1), (1, 1), 1, False, image_size)
        self._bn2 = BatchNormalization()
        
        self.adaptive_avg_pool2d = GlobalAveragePooling2D()
        self._swish = tf.keras.activations.swish
        self._sigmoid = tf.keras.activations.sigmoid
        if self.dropout_rate > 0:
            self.dropout = Dropout(self.dropout_rate, noise_shape=(None, 1, 1, 1))
        else:
            self.dropout = tf.identity
        
    def call(self, inputs):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = self.adaptive_avg_pool2d(x)
            x_squeezed = tf.expand_dims(x_squeezed, -2)
            x_squeezed = tf.expand_dims(x_squeezed, -2)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = self._sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        if self.id_skip and self.strides == 1 and self.input_filters == self.output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            # if drop_connect_rate:
            #     x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = self.dropout(x)
            x = x + inputs  # skip connection
        return x

class ConvHead(Layer):
    def __init__(self, filters, kernel_size, stride, groups, use_bias, image_size, dropout_rate, classes=1000, include_top=True):
        super().__init__()
        self.include_top = include_top
        self._conv_head = Conv2dStaticSamePadding(filters, kernel_size, stride, groups, use_bias, image_size)
        self._bn1 = BatchNormalization()
        self._swish = tf.keras.activations.swish
        
        if self.include_top:
            self._avg_pooling = GlobalAveragePooling2D()
            self._dropout = Dropout(dropout_rate)
            self._fc = Dense(
                classes,
                use_bias=True,
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                activation='softmax'
            )
    
    def call(self, inputs):
        x = self._conv_head(inputs)
        x = self._bn1(x)
        x = self._swish(x)
        if self.include_top:
            x = self._avg_pooling(x)
            x = self._dropout(x)
            x = self._fc(x)
        return x

