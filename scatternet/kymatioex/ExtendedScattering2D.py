import numpy as np
import pywt
from scipy.fft import fft2, ifft2
import tensorflow as tf
from kymatio.keras import Scattering2D
from scatternet.kymatioex.scattering_models import angle_avg_scattering

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
