import numpy as np
import pywt
from scipy.fft import fft2, ifft2
import tensorflow as tf
from kymatio.keras import Scattering2D
from scatternet.kymatioex.scattering_models import angle_avg_scattering
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet

#from utils import FFT
#from LiSA.modules.util.wavelets import StarletTransform
from scatternet.utils.iuwt import iuwt_decomposition

'''
A subclass of the kymatio Scattering2D designed to easily enable extensions in tensorflow.
Explicitly combines the class interface with the tensorflow backend. 
'''
class ExtendedScattering2D(Scattering2D):
  def __init__(self, J, L=8, max_order=2, pre_pad=False): 
        Scattering2D.__init__(self, J, L, max_order, pre_pad)  

  def labels(self, input):

        return self._scattering_func(input, pad = self.S.pad, unpad = self.S.unpad, backend = self.S.backend,
                                     J = self.S.J, L = self.S.L, phi = self.S.phi, psi = self.S.psi,
                                     max_order = self.S.max_order, out_type = "labels", verbose = False)

  def _scattering_func(self, x, **kwargs):
      raise NotImplementedError

  def scattering(self, input):
      with tf.name_scope('scattering') as scope:
          try:
              input = tf.convert_to_tensor(input)
          except ValueError:
              raise TypeError('The input should be convertible to a '
                              'TensorFlow Tensor.')

          if len(input.shape) < 2:
              raise RuntimeError('Input tensor should have at least two '
                                 'dimensions.')

          if (input.shape[-1] != self.shape[-1] or input.shape[-2] != self.shape[-2]) and not self.pre_pad:
              raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.shape[0], self.shape[1]))

          if (input.shape[-1] != self.S._N_padded or input.shape[-2] != self.S._M_padded) and self.pre_pad:
              raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self.S._M_padded, self.S._N_padded))

          if not self.out_type in ('array', 'list'):
              raise RuntimeError("The out_type must be one of 'array' or 'list'.")

          # Use tf.shape to get the dynamic shape of the tf.Tensors at
          # execution time.
          batch_shape = tf.shape(input)[:-2]
          signal_shape = tf.shape(input)[-2:]

          # NOTE: Cannot simply concatenate these using + since they are
          # tf.Tensors and that would add their values.
          input = tf.reshape(input, tf.concat(((-1,), signal_shape), 0))

          S = self._scattering_func(input, pad = self.S.pad, unpad = self.S.unpad, backend = self.S.backend,
                                     J = self.S.J, L = self.S.L, phi = self.S.phi, psi = self.S.psi,
                                     max_order = self.S.max_order, out_type = self.S.out_type)

          if self.out_type == 'array':

              scattering_shape = tf.shape(S)[-3:]
              new_shape = tf.concat((batch_shape, scattering_shape), 0)

              print(scattering_shape, new_shape)
              S = tf.reshape(S, new_shape)

          else:
              scattering_shape = tf.shape(S[0]['coef'])[-2:]
              new_shape = tf.concat((batch_shape, scattering_shape), 0)

              for x in S:
                  x['coef'] = tf.reshape(x['coef'], new_shape)

          return S

class ReducedMorletScattering2D(ExtendedScattering2D):
    def __init__(self, J, L=8, max_order=2, pre_pad=False): 
        ExtendedScattering2D.__init__(self, J, L, max_order, pre_pad)

    def _scattering_func(self, x, **kwargs):
      return angle_avg_scattering(x, **kwargs)

class StarletScattering2D(ExtendedScattering2D):
    def __init__(self, J, max_order=2, pre_pad=False): 
        ExtendedScattering2D.__init__(self, J, 1, max_order, pre_pad)
        self.J = J

    def labels(self):
      return [j for j in range(self.J + self.J*self.J)]

    def _scattering_func(self, x, **kwargs):
      out_S = []
      out_S_0 = []
      out_S_1 = []
      #for n in range(self.S.max_order):
      order0 = iuwt_decomposition(x,self.J)
      out_S_0.append(order0)
      for j in range(self.J):
        order1 = iuwt_decomposition(order0[j,:,:],self.J)
        out_S_1.append(order1)
      out_S.extend(out_S_0)
      out_S.extend(out_S_1)

      out_S = np.concatenate([s for s in out_S])

      #starlet = StarletTransform(num_procs=1)
      #coeffs = starlet.decompose(x,self.J)
      return out_S

    def predict(self, x):
      filters = np.stack([self._scattering_func(x[i,:,:]) for i in range(x.shape[0])],0)
      print(filters.shape)
      return filters

class ShapeletBasis(ShapeletSet):
    def __init__(self): 
        ShapeletSet.__init__(self)   

    def getbasis(self, x, y, n_max, beta, deltaPix, center_x=0, center_y=0):
        """
        decomposes an image into the shapelet coefficients in same order as for the function call
        :param x:
        :param y:
        :param n_max:
        :param beta:
        :param center_x:
        :param center_y:
        :return:
        """
        num_param = int((n_max+1)*(n_max+2)/2)
        print(len(x), len(y))
        base_list = np.zeros( (num_param, len(x)))
        amp_norm = 1./beta**2*deltaPix**2
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': amp_norm}
            base = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            base_list[i,:] = base
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return base_list

class ShapeletScattering2D(ExtendedScattering2D):
    def __init__(self,  beta, n_max, max_order=2, pre_pad=False): 
        ExtendedScattering2D.__init__(self, 1, 1, max_order, pre_pad)
        self.beta = beta
        self.n_max = n_max

    def build(self, input_shape):
        import lenstronomy.Util.util as util
        numPix = input_shape[-1]
        self.x_grid, self.y_grid = util.make_grid(numPix=numPix, deltapix=1)  # make a coordinate grid
        ExtendedScattering2D.build(self, input_shape)
        self.filters = self.create_filters(input_shape)

    def create_filters(self, input_shape):
        shapeletSet = ShapeletBasis()
        basis= shapeletSet.getbasis(self.x_grid, self.y_grid, self.n_max, self.beta, 1., center_x=0, center_y=0) 
        basis = basis.reshape(-1,input_shape[-2], input_shape[-1])
        return basis

    def labels(self):
      out_S = []
      out_S_0 = ['0']
      out_S_1 = []
      out_S_2 = []

      for i, b1 in enumerate(self.filters):
        out_S_1.append('{0}'.format(i+1))
        for j, b2 in enumerate(self.filters):
          out_S_1.append('{0}-{1}'.format(i+1,j+1))
      out_S.extend(out_S_0)
      out_S.extend(out_S_1)
      out_S.extend(out_S_2)
      out_S = [s for s in out_S]
      return out_S

    def _scattering_func(self, x, **kwargs):
      out_S = []
      out_S_0 = [ x]
      out_S_1 = []
      out_S_2 = []

      for i, b1 in enumerate(self.filters):
        order1 = b1*x
        out_S_1.append(order1)
        for j, b2 in enumerate(self.filters):
          order2 = b2*order1
          out_S_2.append(order2)
      out_S.extend(out_S_0)
      out_S.extend(out_S_1)
      out_S.extend(out_S_2)
      out_S = tf.stack([s for s in out_S],-3)
      #out_S = tf.stack([s for s in out_S],-1)
      return out_S

    def predict(self, x):
      filters = np.stack([self._scattering_func(x[i,:,:]) for i in range(x.shape[0])],0)
      return filters
