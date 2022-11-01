import tensorflow as tf
from .utils import (
    round_repeats,
    round_filters,
    calculate_output_image_size
)
from .Components import (
    StemConv,
    MBConvBlock,
    ConvHead
)
from .args import params_dict

def block(x, block_arg, no_block, num_blocks, dropout_rate, image_size):
    filters_in, filters_out, repeats, kernel_size, strides, expand_ratio, id_skip, has_se = block_arg
    
    for j in range(repeats):
        image_size, drop_connect_rate = [image_size, dropout_rate * no_block / num_blocks]
        # print(no_block, filters_in, filters_out, kernel_size, strides, expand_ratio, id_skip, has_se, image_size, drop_connect_rate)
        # The first block needs to take care of stride and filter size increase.
        if j > 0:
            strides, filters_in = (1, 1), filters_out
        x = MBConvBlock(
            filters_in=filters_in, 
            filters_out=filters_out, 
            kernel_size=kernel_size, 
            strides=strides, 
            expand_ratio=expand_ratio, 
            id_skip=id_skip,
            has_se=has_se,
            drop_connect_rate=drop_connect_rate,
            image_size=image_size
        )(x)
        # print()
        image_size = calculate_output_image_size(image_size, strides)
        no_block += 1
    return no_block, image_size, x

def EfficientNet(model_name='efficientnet-b0', include_top=True, classes=1000):
    width_coef, depth_coef, resolution, dropout_rate = params_dict['efficientnet-b0']
    
    inputs = tf.keras.layers.Input((None, None, 3))
    x = inputs
    
    # Conv2dStaticSamePadding
    filters = round_filters(32, width_coefficient=width_coef)
    x = StemConv(
        filters=filters, 
        kernel_size=(3, 3), 
        stride=(2, 2), 
        groups=1, 
        use_bias=False, 
        image_size=resolution
    )(x)
    resolution = calculate_output_image_size(resolution, (2, 2))
    
    # 7個block各重複幾次
    repeat_list = [1, 2, 2, 3, 3, 4, 1]
    num_blocks = float(sum(round_repeats(repeat, depth_coef) for repeat in repeat_list))
    no_block = 0
    
    """
    filters_in, filters_out, repeats, 
    kernel_size, strides, expand_ratio, id_skip, has_se
    """
    # MBConv Block1
    block_arg = [round_filters(32, width_coefficient=width_coef),
                 round_filters(16, width_coefficient=width_coef),
                 round_repeats(1, depth_coef),
                 (3, 3), (1, 1), 1, True, True]
    no_block, resolution, x = block(x, block_arg, no_block, num_blocks, dropout_rate, resolution)
    
    # MBConv Block2
    block_arg = [round_filters(16, width_coefficient=width_coef),
                 round_filters(24, width_coefficient=width_coef),
                 round_repeats(2, depth_coef),
                 (3, 3), (2, 2), 6, True, True]
    no_block, resolution, x = block(x, block_arg, no_block, num_blocks, dropout_rate, resolution)
    
    # MBConv Block3
    block_arg = [round_filters(24, width_coefficient=width_coef),
                 round_filters(40, width_coefficient=width_coef),
                 round_repeats(2, depth_coef),
                 (5, 5), (2, 2), 6, True, True]
    no_block, resolution, x = block(x, block_arg, no_block, num_blocks, dropout_rate, resolution)
    
    # MBConv Block4
    block_arg = [round_filters(40, width_coefficient=width_coef),
                 round_filters(80, width_coefficient=width_coef),
                 round_repeats(3, depth_coef),
                 (3, 3), (2, 2), 6, True, True]
    no_block, resolution, x = block(x, block_arg, no_block, num_blocks, dropout_rate, resolution)
    
    # MBConv Block5
    block_arg = [round_filters(80, width_coefficient=width_coef),
                 round_filters(112, width_coefficient=width_coef),
                 round_repeats(3, depth_coef),
                 (5, 5), (1, 1), 6, True, True]
    no_block, resolution, x = block(x, block_arg, no_block, num_blocks, dropout_rate, resolution)
    
    # MBConv Block6
    block_arg = [round_filters(112, width_coefficient=width_coef),
                 round_filters(192, width_coefficient=width_coef),
                 round_repeats(4, depth_coef),
                 (5, 5), (2, 2), 6, True, True]
    no_block, resolution, x = block(x, block_arg, no_block, num_blocks, dropout_rate, resolution)
    
    # MBConv Block7
    block_arg = [round_filters(192, width_coefficient=width_coef),
                 round_filters(320, width_coefficient=width_coef),
                 round_repeats(1, depth_coef),
                 (3, 3), (1, 1), 6, True, True]
    no_block, resolution, x = block(x, block_arg, no_block, num_blocks, dropout_rate, resolution)
    
    # 
    filters = round_filters(1280, width_coefficient=width_coef)
    resolution = calculate_output_image_size(resolution, (1,1))
    # filters, kernel_size, stride, groups, use_bias, image_size
    outputs = ConvHead(
        filters, 
        kernel_size=(1,1), 
        stride=(1,1), 
        groups=1, 
        use_bias=False, 
        image_size=resolution, 
        dropout_rate=dropout_rate, 
        classes=classes, 
        include_top=include_top
    )(x)
    return  tf.keras.models.Model(inputs=inputs, outputs=outputs)