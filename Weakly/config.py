import numpy as np
import os

class Config(object):
    def __init__(self, args):
        self.lr = eval(args.Weakly_lr)
        self.lr_str = args.Weakly_lr

    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')

