import numpy as np
import tensorflow as tf
from .base2d import ExtendedScattering2D

class ReducedMorletScattering2D(ExtendedScattering2D):
    def __init__(self, J, L=8, max_order=2, subsample = True, pre_pad=False): 
        ExtendedScattering2D.__init__(self, J, L, max_order, pre_pad)
        self.subsample = subsample
        self.lowpass = subsample

    def _scattering_func(self, x, **kwargs):
      return angle_avg_scattering(x, self.subsample, self.lowpass, **kwargs)


def angle_avg_scattering(x, subsample, lowpass, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array', verbose = False):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft    
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    mean = tf.reduce_mean

    # for n1 in range(len(psi)):
    #     print(psi[n1]['j'],psi[n1]['theta'], len(psi[n1]['levels']))
    #     for l in psi[n1]['levels']:
    #         print(l.shape)

    # Define lists for output.
    out_S_0, out_S_1, out_S_2 = [], [], []

    U_r = pad(x)

    U_0_c = rfft(U_r)

    # First low pass filter
    
    if lowpass:
        U_1_c = cdgmm(U_0_c, phi['levels'][0])
    else:
        U_1_c = U_0_c
    if subsample:
        U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    S_0 = irfft(U_1_c)
    S_0 = unpad(S_0)

    # x= S_0
    # U_r = pad(x)
    # U_0_c = rfft(U_r)
    # print(U_0_c.shape)#


    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': (),
                    'label': 'Order 0 low-pass'})

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1]['levels'][0])
        if j1 > 0 and subsample:
            #U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
            U_1_c = subsample_fourier(U_1_c, k=2 ** (j1))
        U_1_c = ifft(U_1_c)
        U_1_c = modulus(U_1_c)
        U_1_c = rfft(U_1_c)

        # Second low pass filter
        if lowpass:
            if subsample:
                S_1_c = cdgmm(U_1_c, phi['levels'][j1])
                S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))
            else:
                S_1_c = cdgmm(U_1_c, phi['levels'][0])
        else:
            S_1_c = U_1_c

        S_1_r = irfft(S_1_c)
        S_1_r = unpad(S_1_r)

        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'n': (n1,),
                        'theta': (theta1,),
                        'label':"Order 1 j={0},l={1}".format(j1,theta1)})

        if max_order < 2: continue

        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1:
                continue


            if subsample:
                U_2_c = cdgmm(U_1_c, psi[n2]['levels'][j1])
                U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            else:
                U_2_c = cdgmm(U_1_c, psi[n2]['levels'][0])

            U_2_c = ifft(U_2_c)
            U_2_c = modulus(U_2_c)
            U_2_c = rfft(U_2_c)

            # Third low pass filter
            if lowpass:
                if subsample:
                    S_2_c = cdgmm(U_2_c, phi['levels'][j2])
                    S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))
                else:
                    S_2_c = cdgmm(U_2_c, phi['levels'][0])
            else:
                S_2_c = U_2_c

            S_2_r = irfft(S_2_c)
            S_2_r = unpad(S_2_r)

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'n': (n1, n2),
                            'theta': (theta1, theta2),
                            'label':"Order 2 j={0},{1},l={2},{3}".format(j1,j2,theta1,theta2)})

    out_S = []
    out_S.extend(out_S_0)

    filtered_out_S_1=[]
    filtered_out_S_2=[]
    for j1 in range(J):
        #if j1 == 0: continue
        combined_1 = [x['coef'] for x in out_S_1 if x['j'][0] == j1]
        combined_1 = tf.stack(combined_1,axis=3)
        combined_1 = tf.reduce_mean(combined_1,axis=-1)
        filtered_out_S_1.append({'coef':combined_1, 'j': (j1,),  'theta':'mean', 'label':"Order 1 j={0}, avg l".format(j1)})
        
        if max_order < 2:
            continue

        for j2 in range(J):
            #if j2 == 0: continue
            if j2 <= j1:
                continue
            combined_2 = [x['coef'] for x in out_S_2 if x['j'] == (j1,j2)]
            combined_2 = tf.stack(combined_2,axis=3)
            combined_2 = tf.reduce_mean(combined_2,axis=-1)
            filtered_out_S_2.append({'coef':combined_2,
                                     'j': (j1, j2),
                                      'theta':'mean',
                                      'label':"Order 2 j={0},{1}, avg l".format(j1,j2)})

            angle_same = [x['coef'] for x in out_S_2 if x['j'] == (j1,j2) and x['theta'][0] == x['theta'][1]]
            angle_m1   = [x['coef'] for x in out_S_2 if x['j'] == (j1,j2) and x['theta'][0] == x['theta'][1]-1]
            angle_p1   = [x['coef'] for x in out_S_2 if x['j'] == (j1,j2) and x['theta'][0] == x['theta'][1]+1]
            angle_same = tf.stack(angle_same,axis=3)
            angle_m1   = tf.stack(angle_m1,axis=3)
            angle_p1   = tf.stack(angle_p1,axis=3)
            angle_same = tf.reduce_mean(angle_same,axis=-1)
            angle_m1   = tf.reduce_mean(angle_m1,axis=-1)
            angle_p1   = tf.reduce_mean(angle_p1,axis=-1)
            filtered_out_S_2.append({'coef':angle_same,'j': (j1, j2), 'theta':'same',  'label':"Order 2 j={0},{1}, l1=l2".format(j1,j2)})
            filtered_out_S_2.append({'coef':angle_m1,  'j': (j1, j2), 'theta':'l1=l2-2',  'label':"Order 2 j={0},{1}, l1=l2-2".format(j1,j2)})
            filtered_out_S_2.append({'coef':angle_p1,  'j': (j1, j2), 'theta':'l1=l2+2',  'label':"Order 2 j={0},{1}, l1=l2+2".format(j1,j2)})
            

    out_S.extend(filtered_out_S_1)
    out_S.extend(filtered_out_S_2)
    #out_S.extend(out_S_1)
    #out_S.extend(out_S_2)

    if verbose:
        for x in out_S: print(x)

    if out_type == 'array':
        out_S = concatenate([x['coef'] for x in out_S])
    if out_type == 'labels':
        out_S = [ x['label'] for x in out_S]

    return out_S