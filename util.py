# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-05-30 00:26:47
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-05-30 00:30:23

import constant as con
import platform


def echo(color, *args):
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
