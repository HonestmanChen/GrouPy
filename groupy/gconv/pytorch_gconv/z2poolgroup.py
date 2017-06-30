import torch.nn as nn


class AvgPoolGroup(nn.AvgPool3d):

    def __init__(self, kernel_size, stride=None):
        super(AvgPoolGroup, self).__init__(kernel_size)
        self.kernel_size = (self.input_stabilizer_size, kernel_size, kernel_size)
        self.stride = (1, stride, stride) if stride else None


class MaxPoolGroup(nn.MaxPool3d):

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPoolGroup, self).__init__(kernel_size)
        self.kernel_size = (self.input_stabilizer_size, kernel_size, kernel_size)
        self.stride = (1, stride, stride) if stride else kernel_size
        self.padding = padding


class Z2AvgPoolP4(AvgPoolGroup):

    @property
    def input_stabilizer_size(self):
        return 4

    @property
    def output_stabilizer_size(self):
        return 1


class Z2AvgPoolP4M(AvgPoolGroup):

    @property
    def input_stabilizer_size(self):
        return 8

    @property
    def output_stabilizer_size(self):
        return 1


class Z2MaxPoolP4(MaxPoolGroup):
    @property
    def input_stabilizer_size(self):
        return 4

    @property
    def output_stabilizer_size(self):
        return 1


class Z2MaxPoolP4M(MaxPoolGroup):
    @property
    def input_stabilizer_size(self):
        return 8

    @property
    def output_stabilizer_size(self):
        return 1
