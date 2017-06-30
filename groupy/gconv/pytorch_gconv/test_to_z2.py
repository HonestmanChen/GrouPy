from groupy.gconv.pytorch_gconv.p4m_conv import P4MConvZ2, P4MConvP4M, Z2ConvP4M
from groupy.gconv.pytorch_gconv.z2poolgroup import Z2AvgPoolP4, Z2AvgPoolP4M, Z2MaxPoolP4, Z2MaxPoolP4M
from groupy.gconv.pytorch_gconv.p4_conv import P4ConvZ2, P4ConvP4, Z2ConvP4
from groupy.gfunc import Z2FuncArray, P4FuncArray, P4MFuncArray
from groupy.garray.C4_array import C4
from groupy.garray.D4_array import D4
from torch.autograd import Variable
from torch import from_numpy
import numpy as np

import unittest


class CheckToZ2Equivariance:
    """
    Parent unittest class containing all test-cases
    relevant for both C4 and D4 subclasses. tests run
    on both convolution and pooling operations
    """

    # TODO: Index out of bounds error for `even' input, in groupy.gfunc.gfuncarray
    # def test_even_input_odd_kernel(self):
    #     ksize = 3
    #
    #     image = np.random.randn(1, 1, 32, 32).astype('float32')
    #     self.perform_z2_to_z2_convolution(image, ksize)
    #
    #     fm = np.random.randn(1, 1, len(self.group), 64, 64).astype('float32')
    #     self.perform_group_to_z2_convolution(fm, ksize)

    # TODO: Even kernel: not yet implemented in GrouPy master
    # def test_odd_input_even_kernel(self):
    #     ksize = 2
    #
    #     image = np.random.randn(1, 1, 9, 9).astype('float32')
    #     self.perform_z2_to_z2_convolution(image, ksize)
    #
    #     fm = np.random.randn(1, 1, len(self.group), 9, 9).astype('float32')
    #     self.perform_group_to_z2_convolution(fm, ksize)

    # TODO: Even kernel: not yet implemented in GrouPy master
    # def test_even_input_even_kernel(self):
    #     ksize = 2
    #
    #     image = np.random.randn(1, 1, 32, 32).astype('float32')
    #     self.perform_z2_to_z2_convolution(image, ksize)
    #
    #     fm = np.random.randn(1, 1, len(self.group), 8, 8).astype('float32')
    #     self.perform_group_to_z2_convolution(fm, ksize)

    def __init__(self):
        self.group = None
        self.GConvZ2 = None
        self.GConvG = None
        self.Z2ConvG = None
        self.Z2PoolG = None
        self.Garray = None

    def test_odd_input_odd_kernel(self):
        """
        Test equivariance using the pooling operation.
        Currently only odd input and kernel sizes work.
        """

        ksize = 3

        image = np.random.randn(1, 1, 9, 9).astype('float32')
        self.perform_z2_to_z2(image, ksize)

        fm = np.random.randn(1, 1, len(self.group), 9, 9).astype('float32')
        self.perform_group_to_z2(fm, ksize)

    def check_equivariance(self, im, layers, input_array, output_array):
        """
        Performs equivariance check on given input, by using all transformations
        of group subclass.
        """
        image = Variable(from_numpy(im))

        for transformation in self.group:
            image_t = (transformation * input_array(im)).v
            image_t = Variable(from_numpy(image_t))

            # Perform convolutions
            fm_image = image
            fm_image_t = image_t
            for layer in layers:
                fm_image = layer(fm_image)
                fm_image_t = layer(fm_image_t)

            # Transform output to prove equivariance
            fm_image = fm_image.data.numpy()
            fm_image_t = output_array(fm_image_t.data.numpy())
            fm_image_t = (transformation.inv() * fm_image_t).v

            self.assert_equal_output(fm_image, fm_image_t)

    @staticmethod
    def assert_equal_output(out1, out2, rtol=1e-5, atol=1e-3):
        assert np.allclose(out1, out2, rtol=rtol, atol=atol), \
               'Equivariance property does not hold'

    def perform_z2_to_z2(self, image, ksize):
        raise NotImplementedError

    def perform_group_to_z2(self, fm, ksize):
        raise NotImplementedError


class CheckZ2ConvGroupEquivariance(CheckToZ2Equivariance):

    def perform_z2_to_z2(self, im, ksize):
        self.check_equivariance(
            im=im,
            layers=[
                self.GConvZ2(in_channels=1, out_channels=2, ksize=ksize),
                self.GConvG(in_channels=2, out_channels=3, ksize=ksize),
                self.Z2ConvG(in_channels=3, out_channels=1, ksize=ksize)
            ],
            input_array=Z2FuncArray,
            output_array=Z2FuncArray
        )

    def perform_group_to_z2(self, fm, ksize):
        self.check_equivariance(
            im=fm,
            layers=[
                self.GConvG(in_channels=1, out_channels=3, ksize=ksize),
                self.Z2ConvG(in_channels=3, out_channels=1, ksize=ksize)
            ],
            input_array=self.Garray,
            output_array=Z2FuncArray
        )


class CheckZ2PoolGroupEquivariance(CheckToZ2Equivariance):

    def perform_z2_to_z2(self, im, ksize):
        self.check_equivariance(
            im=im,
            layers=[
                self.GConvZ2(in_channels=1, out_channels=2, ksize=ksize),
                self.GConvG(in_channels=2, out_channels=3, ksize=ksize),
                self.Z2PoolG(kernel_size=1)
            ],
            input_array=Z2FuncArray,
            output_array=Z2FuncArray
        )

    def perform_group_to_z2(self, fm, ksize):
        self.check_equivariance(
            im=fm,
            layers=[
                self.GConvG(in_channels=1, out_channels=3, ksize=ksize),
                self.Z2PoolG(kernel_size=1)
            ],
            input_array=self.Garray,
            output_array=Z2FuncArray
        )


class CheckP4ConvEquivariance(unittest.TestCase, CheckZ2ConvGroupEquivariance):
    def setUp(self):
        self.group   = C4
        self.GConvZ2 = P4ConvZ2
        self.GConvG  = P4ConvP4
        self.Z2ConvG = Z2ConvP4
        self.Garray  = P4FuncArray


class CheckP4MConvEquivariance(unittest.TestCase, CheckZ2ConvGroupEquivariance):
    def setUp(self):
        self.group   = D4
        self.GConvZ2 = P4MConvZ2
        self.GConvG  = P4MConvP4M
        self.Z2ConvG = Z2ConvP4M
        self.Garray  = P4MFuncArray


class CheckP4AvgPoolEquivariance(unittest.TestCase, CheckZ2PoolGroupEquivariance):
    def setUp(self):
        self.group   = C4
        self.GConvZ2 = P4ConvZ2
        self.GConvG  = P4ConvP4
        self.Z2PoolG = Z2AvgPoolP4
        self.Garray  = P4FuncArray


class CheckP4MAvgPoolEquivariance(unittest.TestCase, CheckZ2PoolGroupEquivariance):
    def setUp(self):
        self.group   = D4
        self.GConvZ2 = P4MConvZ2
        self.GConvG  = P4MConvP4M
        self.Z2PoolG = Z2AvgPoolP4M
        self.Garray  = P4MFuncArray


class CheckP4MaxPoolEquivariance(unittest.TestCase, CheckZ2PoolGroupEquivariance):
    def setUp(self):
        self.group   = C4
        self.GConvZ2 = P4ConvZ2
        self.GConvG  = P4ConvP4
        self.Z2PoolG = Z2MaxPoolP4
        self.Garray  = P4FuncArray


class CheckP4MMaxPoolEquivariance(unittest.TestCase, CheckZ2PoolGroupEquivariance):
    def setUp(self):
        self.group   = D4
        self.GConvZ2 = P4MConvZ2
        self.GConvG  = P4MConvP4M
        self.Z2PoolG = Z2MaxPoolP4M
        self.Garray  = P4MFuncArray


if __name__ == '__main__':
    unittest.main()
