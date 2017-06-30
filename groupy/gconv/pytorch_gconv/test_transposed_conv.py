from groupy.gconv.pytorch_gconv.p4m_conv import P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.p4_conv import P4ConvZ2, P4ConvP4
from groupy.gfunc import Z2FuncArray, P4FuncArray, P4MFuncArray
from groupy.garray.C4_array import C4
from groupy.garray.D4_array import D4
from torch.autograd import Variable
import numpy as np
from torch import from_numpy
import unittest


class CheckTransposedConvolution:
    """
    Parent unittest class containing all test-cases
    relevant for both C4 and D4 subclasses. Tests run
    on multiple stride sizes to check for equivariance
    and output sizes.
    """

    # TODO: Index out of bounds error for `even' input, in groupy.gfunc.gfuncarray
    # def test_even_input_odd_kernel(self):
    #     ksize = 3
    #
    #     image = np.random.randn(1, 1, 32, 32).astype('float32')
    #     self.perform_z2_to_group(image, ksize)
    #     self.perform_z2_to_group(image, ksize, stride=3)
    #
    #     fm = np.random.randn(1, 1, len(self.group), 64, 64).astype('float32')
    #     self.perform_group_to_group(fm, ksize)
    #     self.perform_group_to_group(fm, ksize, stride=3)

    # TODO: Even kernel: not yet implemented in GrouPy master
    # def test_odd_input_even_kernel(self):
    #     ksize = 2
    #
    #     image = np.random.randn(1, 1, 9, 9).astype('float32')
    #     self.perform_z2_to_group(image, ksize)
    #     self.perform_z2_to_group(image, ksize, stride=3)
    #
    #     fm = np.random.randn(1, 1, len(self.group), 9, 9).astype('float32')
    #     self.perform_group_to_group(fm, ksize)
    #     self.perform_group_to_group(fm, ksize, stride=3)

    # TODO: Even kernel: not yet implemented in GrouPy master
    # def test_even_input_even_kernel(self):
    #     ksize = 2
    #
    #     image = np.random.randn(1, 1, 32, 32).astype('float32')
    #     self.perform_z2_to_group(image, ksize)
    #     self.perform_z2_to_group(image, ksize, stride=3)
    #
    #     fm = np.random.randn(1, 1, len(self.group), 8, 8).astype('float32')
    #     self.perform_group_to_group(fm, ksize)
    #     self.perform_group_to_group(fm, ksize, stride=3)

    def __init__(self):
        self.group = None
        self.GConvZ2 = None
        self.GConvG = None
        self.Z2ConvG = None
        self.Z2PoolG = None
        self.Garray = None

    def test_odd_input_odd_kernel(self):
        """
        Test equivariance using the transposed convolution operation.
        Currently only odd input and kernel sizes work.
        """
        ksize = 3

        image = np.random.randn(1, 1, 3, 3).astype('float32')
        self.perform_z2_to_group(image, ksize)
        self.perform_z2_to_group(image, ksize, stride=3)

        fm = np.random.randn(1, 1, len(self.group), 9, 9).astype('float32')
        self.perform_group_to_group(fm, ksize)
        self.perform_group_to_group(fm, ksize, stride=3)

    def test_odd_input_odd_kernel_extremely_strided(self):
        """
        Test equivariance using the transposed convolution operation.
        Using extreme stride value.
        """
        ksize = 3

        image = np.random.randn(1, 1, 3, 3).astype('float32')
        self.perform_z2_to_group(image, ksize, stride=10)

        fm = np.random.randn(1, 1, len(self.group), 3, 3).astype('float32')
        self.perform_group_to_group(fm, ksize, stride=10)

    def test_output_size_stride_one(self):
        """
        Tests output size of transposed convolution operation
        for stride = 1.
        """
        stride = 1
        image = np.random.randn(1, 1, 9, 9).astype('float32')
        image = Variable(from_numpy(image))
        layer = self.GConvZ2(1, 1, ksize=3, stride=stride, transposed=True)
        output = layer(image)
        self.assertEqual(list(output.size()), [1, 1, len(self.group), 9+2*stride, 9+2*stride])

    def test_strided_output_size(self):
        """
        Tests output size of transposed convolution operation
        for a stride bigger then one.
        """
        stride = 3
        image = np.random.randn(1, 1, 3, 3).astype('float32')
        image = Variable(from_numpy(image))
        layer = self.GConvZ2(1, 1, ksize=3, stride=stride, transposed=True)
        output = layer(image)
        self.assertEqual(list(output.size()), [1, 1, len(self.group), 3+2*stride, 3+2*stride])

    def perform_z2_to_group(self, im, ksize, stride=1):
        self.check_equivariance(
            im=im,
            layers=[
                self.GConvZ2(in_channels=1, out_channels=2, ksize=ksize, transposed=True, stride=stride),
                self.GConvG(in_channels=2, out_channels=3, ksize=ksize, transposed=True)
            ],
            input_array=Z2FuncArray,
            output_array=self.Garray
        )

    def perform_group_to_group(self, fm, ksize, stride=1):
        self.check_equivariance(
            im=fm,
            layers=[self.GConvG(in_channels=1, out_channels=3, ksize=ksize, transposed=True, stride=stride)],
            input_array=self.Garray,
            output_array=self.Garray
        )

    def check_equivariance(self, im, layers, input_array, output_array):
        """
        Performs equivariance check on given input, by using all transformations
        of `group' subclass.
        """
        image = Variable(from_numpy(im))

        for transformation in self.group:
            image_t = (transformation * input_array(im)).v
            image_t = Variable(from_numpy(image_t))

            # perform convolutions
            fm_image = image
            fm_image_t = image_t
            for layer in layers:
                fm_image = layer(fm_image)
                fm_image_t = layer(fm_image_t)

            # transform output to prove equivariance
            fm_image = fm_image.data.numpy()
            fm_image_t = output_array(fm_image_t.data.numpy())
            fm_image_t = (transformation.inv() * fm_image_t).v

            self.assert_equal_output(fm_image, fm_image_t)

    @staticmethod
    def assert_equal_output(out1, out2, rtol=1e-5, atol=1e-3):
        assert np.allclose(out1, out2, rtol=rtol, atol=atol), \
               'Equivariance property does not hold'


class CheckP4ConvEquivariance(unittest.TestCase, CheckTransposedConvolution):
    def setUp(self):
        self.group   = C4
        self.GConvZ2 = P4ConvZ2
        self.GConvG  = P4ConvP4
        self.Garray  = P4FuncArray


class CheckP4MConvEquivariance(unittest.TestCase, CheckTransposedConvolution):
    def setUp(self):
        self.group   = D4
        self.GConvZ2 = P4MConvZ2
        self.GConvG  = P4MConvP4M
        self.Garray  = P4MFuncArray


if __name__ == '__main__':
    unittest.main()
