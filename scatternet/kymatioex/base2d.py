import numpy as np
import tensorflow as tf
from kymatio.keras import Scattering2D

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

def standard_scattering(x, pad, unpad, backend, J, L, phi, psi, max_order, out_type='array'):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft    
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate

    # Define lists for output.
    out_S_0, out_S_1, out_S_2 = [], [], []

    U_r = pad(x)

    U_0_c = rfft(U_r)

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi['levels'][0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    S_0 = irfft(U_1_c)
    S_0 = unpad(S_0)

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': ()})

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1]['levels'][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = ifft(U_1_c)
        U_1_c = modulus(U_1_c)
        U_1_c = rfft(U_1_c)

        # Second low pass filter
        S_1_c = cdgmm(U_1_c, phi['levels'][j1])
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))

        S_1_r = irfft(S_1_c)
        S_1_r = unpad(S_1_r)

        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'n': (n1,),
                        'theta': (theta1,)})

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1:
                continue

            U_2_c = cdgmm(U_1_c, psi[n2]['levels'][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = ifft(U_2_c)
            U_2_c = modulus(U_2_c)
            U_2_c = rfft(U_2_c)

            # Third low pass filter
            S_2_c = cdgmm(U_2_c, phi['levels'][j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))

            S_2_r = irfft(S_2_c)
            S_2_r = unpad(S_2_r)

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'n': (n1, n2),
                            'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array':
        out_S = concatenate([x['coef'] for x in out_S])

    elif out_type == 'list':
        out_S = [x['coef'] for x in out_S]

    return out_S

