# -*- encoding: utf-8 -*-
'''
@File    :   LrScheduler.py
@Time    :   2022/11/30 16:53:16
@Author  :   Jim Hsu 
@Version :   1.0
@Contact :   jimhsu11@gmail.com
'''

# here put the import lib
import numpy as np

class ReduceLROnPlateau(object):
    def __init__(
        self, 
        optimizer, 
        factor=0.1, 
        patience=10,
        mode='min', 
        threshold=1e-4, 
        threshold_mode='rel',
        cooldown=0,
        min_lr=0, 
        verbose=True
    ):
        """給定要監控的數值(metrics)，根據是否比之前的好來調整Learning Rate
        也可以給定batch內得出的數值

        Args:
            optimizer (tf.keras.optimizers): 要修改LR的優化器
            factor (float, optional): LR的縮放比例. Defaults to 0.1.
            patience (int, optional): 當模型連續訓練五個epoch後，監控的數值表現沒有更好，就會降低LR. Defaults to 10.
            mode (str, optional): 選擇監控的模式，限定在[min, max]中選擇一種. Defaults to 'min'.
            threshold (float, optional): 衡量新的和過往最佳的監控數值差距，所需加上或減上的閾值. Defaults to 1e-4.
            threshold_mode (str, optional): 和過往最佳的監控數值比較的方法，限定在[rel, abs]中選擇一種. Defaults to 'rel'.
            cooldown (int, optional): 每次更新LR後再次進行監控所需等待的epoch數量. Defaults to 0.
            min_lr (int, optional): 最小的LR數值，到了這個數值LR不會再降低. Defaults to 0.
            verbose (bool, optional): 是否顯示更新LR的訊息. Defaults to True.

        Raises:
            ValueError: factor不在預設範圍內 (>=1)
        """        
        if factor >= 1.0:
            raise ValueError(
                f"ReduceLROnPlateau does not support "
                f"a factor >= 1.0. Got {factor}"
            )
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.verbose = verbose
        self._reset()
    
    def _reset(self):
        if self.mode not in {'min', 'max'}:
            raise ValueError('mode ' + self.mode + ' is unknown!')
        if self.threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + self.threshold_mode + ' is unknown!')
        if self.mode == 'min':
            self.mode_worse = np.Inf
        else:  # mode == 'max':
            self.mode_worse = -np.Inf
        
        self.cooldown_counter = 0
        self.best = self.mode_worse
        self.mode_worse = None  # the worse value for the chosen mode
        self.last_epoch = 0
        self.num_bad_epochs = 0
        
    def step(self, metrics, epoch=None):
        current = metrics
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  
            
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        
        # 冷卻結束
        elif self.cooldown_counter <= 0:
            self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                old_lr = self.optimizer.lr
                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    self.optimizer.lr = new_lr
                    if self.verbose:
                        print(
                            f"\nEpoch {epoch +1}: "
                            f"ReduceLROnPlateau reducing "
                            f"learning rate to {new_lr}."
                        )
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0
    
    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return np.less(a, best * rel_epsilon)

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return np.less(a, best - self.threshold)

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return np.greater(a, best * rel_epsilon)

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return np.greater(a, best + self.threshold)


