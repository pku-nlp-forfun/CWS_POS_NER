# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-05-30 00:26:47
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-05-30 13:12:24

import constant as con
import os
import pickle
import platform
import time


def echo(color:int, *args):
    ''' echo log @param: color: 0 -> red, 1 -> green, 2 -> yellow, 3 -> blue '''
    args = ' '.join([str(ii) for ii in args])
    if con.is_service:
        with open(con.log_path, 'a') as f:
            f.write('{}\n'.format(args))
        return
    colors = {'red': '\033[91m', 'green': '\033[92m',
              'yellow': '\033[93m', 'blue': '\033[94m'}
    if type(color) != int or not color in list(range(len(colors.keys()))) or platform.system() == 'Windows':
        print(args)
    else:
        print(list(colors.values())[color], args, '\033[0m')

def dump_bigger(data, output_file: str):
    ''' pickle.dump big file which size more than 4GB '''
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    with open(output_file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_bigger(input_file: str):
    ''' pickle.load big file which size more than 4GB '''
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)

def time_str(timestamp: int = -1, format: str = '%Y-%m-%d %H:%M:%S') -> str:
    ''' time str '''
    if timestamp > 0:
        return time.strftime(format, time.localtime(timestamp))
    return time.strftime(format, time.localtime(time.time()))

def log(log_str:str):
    ''' log record '''
    with open(con.log_path, 'a') as f:
        f.write(f'{log_str}\n')