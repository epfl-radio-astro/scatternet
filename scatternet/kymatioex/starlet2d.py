import numpy as np
import tensorflow as tf
from .base2d import ExtendedScattering2D
from scatternet.utils.iuwt import iuwt_decomposition

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
      if self.max_order >= 1:
        for j in range(self.J):
          order1 = iuwt_decomposition(order0[j,:,:],self.J)
          out_S_1.append(order1)

      out_S.extend(out_S_0)

      if self.max_order >= 1: out_S.extend(out_S_1)

      out_S = np.concatenate([s for s in out_S])

      #starlet = StarletTransform(num_procs=1)
      #coeffs = starlet.decompose(x,self.J)
      return out_S

    def predict(self, x):
      filters = np.stack([self._scattering_func(x[i,:,:]) for i in range(x.shape[0])],0)
      print(filters.shape)
      return filters