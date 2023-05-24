import numpy as np
import tensorflow as tf
from .base2d import ExtendedScattering2D
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet

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
        """
        :param beta: shapelet scale
        :param n_max: maximum polynomial order in Hermite polynomial
        """

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
        if self.max_order < 2: continue
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
        if self.max_order < 2: continue
        for j, b2 in enumerate(self.filters):
          order2 = b2*order1
          out_S_2.append(order2)
      out_S.extend(out_S_0)
      out_S.extend(out_S_1)
      out_S.extend(out_S_2)
      out_S = tf.stack([s for s in out_S],-3)

      return out_S

    def predict(self, x):
      filters = np.stack([self._scattering_func(x[i,:,:]) for i in range(x.shape[0])],0)
      return filters