# -*- encoding: utf-8 -*-
'''
@File    :   Monitor.py
@Time    :   2022/11/30 16:57:30
@Author  :   Jim Hsu 
@Version :   1.0
@Contact :   jimhsu11@gmail.com
'''

# here put the import lib
import os
import tensorflow as tf
from datetime import datetime
from collections import OrderedDict

class Monitor():
    def __init__(self, exp_name, monitor_name) -> None:
        # save with datetime
        os.makedirs('logs', exist_ok=True)
        ExpDir = os.path.join('logs', exp_name)
        if os.path.exists(os.path.join(ExpDir, monitor_name + '.log')):
            ExpDir += '_' + datetime.now().strftime("%H%M%S")
        os.makedirs(ExpDir, exist_ok=True)
        self.log_path = os.path.join(ExpDir, monitor_name + '.log')
            
        self.watch_dict = OrderedDict()
        self.epoch_dict = {}
        self.max_epoch = 0
    
    def watch(self, variable, variable_name):
        self.add(variable, variable_name)
        
        for name in self.watch_dict.keys():
            self.epoch_dict[name] = OrderedDict()
        
    def add(self, variable, variable_name):
        if isinstance(variable, list) and isinstance(variable_name, list):
            for idx, name in enumerate(variable_name):
                self.watch_dict[name] = variable[idx]
        elif isinstance(variable, str) and isinstance(variable_name, str):
            self.watch_dict[variable_name] = variable
        else:
            raise TypeError("兩者型態必須同為list或是str") 
        
    def show(self):
        for name in self.watch_dict.keys():
            print(name, ': ', self.epoch_dict[name][self.max_epoch])
    
    def batch_show(self):
        update_str = ''
        for name in self.watch_dict.keys():
            update_str += '{}: {:.4f}  '.format(name, self.watch_dict[name].result().numpy())
        print(update_str , "\r" , end='')
    
    def reset_states(self):
        for name in self.watch_dict.keys():
            self.watch_dict[name].reset_states()
    
    def update_state(self, epoch):
        self.max_epoch = epoch
        for name in self.watch_dict.keys():
            self.epoch_dict[name][self.max_epoch] = self.watch_dict[name].result().numpy()

    def save(self):
        if not os.path.exists(self.log_path):
            first_line = ''
            for name in self.epoch_dict.keys():
                first_line += name + ','
            first_line = first_line[:-1] + '\n'
            with open(self.log_path, 'w') as log_writer:
                log_writer.write(first_line)

        with open(self.log_path, 'a') as log_writer:
            line = ''
            for name in self.epoch_dict.keys():
                line += str(self.epoch_dict[name][self.max_epoch]) + ','
            line = line[:-1] + '\n'
            log_writer.write(line)

# TODO 如果要追蹤非tf.keras.metrics，可試著建構'變數class'來記錄
# 在monitor用額外的函式紀錄
class var_object:
    def __init__(self):
        self.var = tf.constant(0.)
    def __repr__(self):
        return str(self.var)
    def __str__(self):
        return str(self.var)
    def result(self):
        return self.var
    def update_state(self, var):
        # NOTE 目前是只紀錄變數，可透過繼承重寫更新方法
        self.var = var
    def reset_states(self):
        self.var = tf.constant(0.)
        
        


