"""
Python implementation of digital biquad filters

reference: https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
"""
import numpy as np
from scipy import signal

def compute_biquad_filter(length, x, y, x_buf, y_buf, b, a):
    """
    a[0] == 1.0
    """
    for k in range(length):
        y[k] = b[0] * x[k]\
             + b[1] * x_buf[1]\
             + b[2] * x_buf[0]\
             - a[1] * y_buf[1]\
             - a[2] * y_buf[0]
        x_buf[0] = x_buf[1]
        x_buf[1] = x[k]
        y_buf[0] = y_buf[1]
        y_buf[1] = y[k]

class PyQuadFilter():
    """ Bi-quad filter on python

    Parameters
    ----------
    rate : float
        Sampling rate  (> 0.0)
    filter_class : str
        - "lowpass"
        - "highpass"
        - "bandpass"
        - "allpass"
        - "notch"
        - "peaking"
        - "lowshelf"
        - "highshelf"
    fc : float
        Cut-off frequency (> 0.0)
    q : float
        q value (> 0.0)
    gain_db : float
        gain in dB
    """
    def __init__(
        self, 
        sr, 
        filter_type=None,
        fc=1000,
        q=1.0,
        gain_db=1.0
    ):
        if sr < 0.0:
            raise ValueError("sr cannot be given a negative value")
        self._sr = sr
    
        self._b = np.zeros(3)
        self._a = np.zeros(3)
        self._y = None
 
        if filter_type is not None:
            self.set_params(filter_type, fc, q, gain_db)


    def set_params(self, filter_type, fc, q, gain_db):
        b, a = self._b.view(), self._a.view()
        b *= 0.0
        a *= 0.0
    
        if fc < 0.0 or fc >= self._sr / 2.0:
            raise ValueError(f"illegal value: fc={fc}")
        self._fc = fc

        if q < 0.0:
            raise ValueError(f"illegal value: q={q}")
        self._q = q
        
        self._gain_db = gain_db
        self.amp = 10.0 ** (gain_db/40.0)
        
        w = 2.0 * np.pi * self._fc / self._sr
        cos_w, sin_w = np.cos(w), np.sin(w)
        alpha = 0.5 * sin_w / self._q
        
        filter_type = filter_type.lower()\
                        .replace("-","").replace("_","")

        if filter_type == "lowpass":
            b[0] = (1.0 - cos_w) * 0.5
            b[1] = (1.0 - cos_w)
            b[2] = (1.0 - cos_w) * 0.5
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
            b /= a[0]
            a /= a[0]
        elif filter_type == "highpass":
            b[0] = (1.0 + cos_w) * 0.5
            b[1] = -(1.0 + cos_w)
            b[2] = (1.0 + cos_w) * 0.5
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
            b /= a[0]
            a /= a[0]
        elif filter_type == "bandpass":
            b[0] =  sin_w * 0.5
            b[1] = 0.0
            b[2] = -sin_w * 0.5
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
            b /= a[0]
            a /= a[0]
        elif filter_type == "allpass":
            b[0] = 1.0 - alpha
            b[1] = -2.0 * cos_w
            b[2] = 1.0 + alpha
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
            b /= a[0]
            a /= a[0]
        elif filter_type == "notch":
            b[0] = 1.0
            b[1] = -2.0 * cos_w
            b[2] = 1.0
            a[0] = 1.0 + alpha
            a[1] = -2.0 * cos_w
            a[2] = 1.0 - alpha
            b /= a[0]
            a /= a[0]
        elif filter_type == "peaking":
            b[0] = 1.0 + alpha * self.amp
            b[1] = -2.0 * cos_w
            b[2] = 1.0 - alpha * self.amp
            with np.errstate(divide='ignore'):
                a[0] = 1.0 + alpha / self.amp
                a[1] = -2.0 * cos_w
                a[2] = 1.0 - alpha / self.amp
            b /= a[0]
            a /= a[0]
        elif filter_type == "lowshelf":
            amp_add_1 = self.amp + 1.0
            amp_sub_1 = self.amp - 1.0
            sqrt_amp  = np.sqrt(self.amp)
            b[0] = self.amp * (amp_add_1 - amp_sub_1 * cos_w\
                 + 2.0 * sqrt_amp * alpha)
            b[1] = 2.0 * self.amp * (amp_sub_1 - amp_add_1 * cos_w)
            b[2] = self.amp * (amp_add_1 - amp_sub_1 * cos_w\
                 - 2.0 * sqrt_amp * alpha)
            a[0] = amp_add_1 + amp_sub_1 * cos_w\
                 + 2.0 * sqrt_amp * alpha
            a[1] = -2.0 * (amp_sub_1 + amp_add_1 * cos_w)
            a[2] = amp_add_1 + amp_sub_1 * cos_w\
                 - 2.0 * sqrt_amp * alpha
            b /= a[0]
            a /= a[0]
        elif filter_type == "highshelf":
            amp_add_1 = self.amp + 1.0
            amp_sub_1 = self.amp - 1.0
            sqrt_amp  = np.sqrt(self.amp)
            
            b[0] = self.amp * (amp_add_1 + amp_sub_1 * cos_w\
                 + 2.0 * sqrt_amp * alpha)
            b[1] = -2.0 * self.amp * (amp_sub_1 + amp_add_1 * cos_w)
            b[2] = self.amp * (amp_add_1 + amp_sub_1 * cos_w\
                 - 2.0 * sqrt_amp * alpha)
            a[0] = amp_add_1 - amp_sub_1 * cos_w\
                 + 2.0 * sqrt_amp * alpha
            a[1] = 2.0 * (amp_sub_1 - amp_add_1 * cos_w)
            a[2] = amp_add_1 - amp_sub_1 * cos_w\
                 - 2.0 * sqrt_amp * alpha
            b /= a[0]
            a /= a[0]
        else:
            raise ValueError(f"invalid filter_type: {filter_type}")
        
        
        self.filter_type = filter_type
        return b, a

    def filter(self, x, online=False):
        if x.ndim == 1:
            x_tmp = x[np.newaxis, :]
        elif x.ndim > 2:
            raise ValueError(f"x.ndim is not 2:  x.ndim=={x.ndim}")
        else: # x.ndim == 2
            x_tmp = x.view()
            
        if online:
            return self._filter_online(x_tmp)
        else:
            return self._filter_offline(x_tmp)
    
    def prepare_filter_online(self, n_ch=None, length=None):
        self._y = np.zeros((n_ch, length))
        self.x_buf   = np.zeros((n_ch, 2))
        self.y_buf   = np.zeros((n_ch, 2))
        
    def _filter_online(self, x):
        if x.shape != self._y.shape:
            raise ValueError(f"x.shape and self._y.shape are not equal: {x.shape}, {self._y.shape}")

        n_ch, n_samples = x.shape
        for k in range(n_ch):
            compute_biquad_filter(n_samples, x[k], self._y[k], 
                                  self.x_buf[k], self.y_buf[k], 
                                  self._b, self._a)
                
        return self._y
        
    def _filter_offline(self, x):
        n_ch, n_samples = x.shape
        y = np.zeros(x.shape)
        x_buf   = np.zeros((n_ch, 2))
        y_buf   = np.zeros((n_ch, 2))

        for k in range(n_ch):
            compute_biquad_filter(n_samples, x[k], y[k], 
                                  x_buf[k], y_buf[k],
                                  self._b, self._a)
                                  
            #if filter_type == "bellallpass":
            #    y[k] = 0.5 * (x[k] * (1.0 + self.amp) + y[k] * (1.0 - self.amp))
            
        return y

    def __str__(self):
        ret = f"""
        type:    {self.filter_type}
        fc:      {self._fc} Hz
        q:       {self._q}
        gain_db: {self._gain_db}
        
        b: {self._b}
        a: {self._a}
        """
        return ret

    @property
    def b(self):
        return self._b

    @property
    def a(self):
        return self._a

    @property
    def coefs(self):
        return self._b, self._a

    @property
    def sr(self):
        return self._sr

    @property
    def q(self):
        return self._q
        
    @property
    def fc(self):
        return self._fc

    @property
    def gain_db(self):
        return self._gain_db

    @property
    def y(self):
        return self._y
