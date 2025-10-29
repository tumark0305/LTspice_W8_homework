import numpy as np
import torch,threading,time,multiprocessing
import matplotlib.pyplot as plt
from scipy.signal import lombscargle

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class calculator:
    _lock = multiprocessing.Lock()
    harmonic_list = [1,2,3,4,5,6,7,8,9,10,11]
    def __init__(self,_time_array,_y_array):
        _got = calculator._lock.acquire(block=True)
        self.x = _time_array
        self.y = _y_array
        self.resonate_detector()
        if self.stop_resonate_time is not None:
            self.resonate_only(1e-3)
        self.fft_result = None
        self._lock.release()
        return None
    def resonate_detector(self):
        self.stop_resonate_time = None
        _x = torch.from_numpy(self.x).to(DEVICE)
        _y = torch.from_numpy(self.y).to(DEVICE)
        _z = (_y - _y.mean()).abs()
        _mid = (_z[1:-1] > _z[:-2]) & (_z[1:-1] >= _z[2:])
        peak_idx = torch.nonzero(_mid, as_tuple=False).squeeze(-1) + 1
        if peak_idx.numel() < 5:
            return None
        A3 = _z[peak_idx[2]].item()
        thr = (A3 * 0.01)
        for i in range(2, peak_idx.numel()-1):
            a1 = _z[peak_idx[i]].item()
            a2 = _z[peak_idx[i+1]].item()
            if a1 >= thr and a2 < thr:
                t1 = _x[peak_idx[i]].item()
                t2 = _x[peak_idx[i+1]].item()
                frac = (thr - a1) / (a2 - a1 + 1e-30)
                t_stop = t1 + frac * (t2 - t1)
                self.stop_resonate_time = float(t_stop)
                break
        if i >=peak_idx.numel()-2:
            self.stop_resonate_time = float(peak_idx.numel())-2
        return None
    def resonate_only(self,_max_duration:float):
        _new_end_time = self.stop_resonate_time
        if _new_end_time>_max_duration:
            _new_end_time=_max_duration
        _mask = self.x < _new_end_time
        self.x = self.x[_mask]
        self.y = self.y[_mask]
        return None
    def FFT(self):
        def fft_core():
            _x = torch.from_numpy(self.x).to(DEVICE)
            _y = torch.from_numpy(self.y).to(DEVICE)
            dt = (_x[1] - _x[0])
            fs = 1.0 / dt
            N  = _y.numel()
            v0   = _y - _y.mean()
            win  = torch.hann_window(N, periodic=True, device=DEVICE, dtype=v0.dtype)
            vwin = v0 * win
            Vf   = torch.fft.rfft(vwin)                 # complex tensor on GPU
            freq = torch.fft.rfftfreq(N, d=float(dt)).to(DEVICE)  # Hz
            amp_scale = 2.0 / (win.sum() / N)
            #mag_db = 20.0 * torch.log10(torch.clamp(mag, min=1e-20))
            mag = torch.abs(Vf) * amp_scale / N
            return freq , mag
        def lomb_fft_core():
            t = self.x - self.x[0]    
            y = self.y - np.mean(self.y) 
            f = np.linspace(1e3, 1e6, int(1e5)) 

            angular_freq = 2 * np.pi * f
            pgram = lombscargle(t, y, angular_freq)
            amp = np.sqrt(4*pgram / len(t))
            return torch.from_numpy(f) , torch.from_numpy(amp)
        def lomb_fft_gpu_core():
            _x = torch.from_numpy(self.x).to(DEVICE).to(torch.float64)
            _y = torch.from_numpy(self.y).to(DEVICE).to(torch.complex64)
            _f = torch.linspace(1e3, 1e7, int(1e5), device=DEVICE, dtype=torch.float64)
            _w = 2 * torch.pi * _f[:, None] *_x[None, :]  # shape (M, N)
            _exp_term = torch.exp((-1j * _w).to(torch.complex64)) 
            _Y = torch.matmul(_exp_term, _y) / _x.numel()
            _mag = torch.abs(_Y)
            return _f , _mag
        def lomb_batched_fft_gpu_core():
            batch_size = int(2.5e4)
            _ctype = torch.complex64
            _x = torch.from_numpy(self.x).to(DEVICE).to(torch.float64)
            _y = torch.from_numpy(self.y).to(DEVICE).to(torch.float64)
            _y_c = torch.from_numpy(self.y).to(DEVICE).to(_ctype)
            _f = torch.linspace(1e5, 1e10, int(1e7), device=DEVICE, dtype=torch.float64)
            _N = _x.numel()
            _Y = torch.zeros((_f.numel(),), device=DEVICE, dtype=_ctype)
            for i in range(0, _f.numel(), batch_size):
                f_chunk = _f[i:i+batch_size]                        # (C,)
                w = 2.0 * torch.pi * f_chunk[:, None] * _x[None, :]      # (C,N) float64
                exp_term = torch.exp((-1j * w).to(_ctype))                 # (C,N) complex
                _Y[i:i+batch_size] = torch.matmul(exp_term, _y_c) / _N            # (C,)
                del w, exp_term
            _mag = torch.abs(_Y)
            return _f , _mag
        
        _freq_array , _mag_list = lomb_batched_fft_gpu_core()

        idx1 = 1 if _freq_array.numel() > 1 else 0
        main_idx = torch.argmax(_mag_list[idx1:]) + idx1
        f_main = _freq_array[main_idx]
        a_main = _mag_list[main_idx]
        self.main_harmonic = []
        for n in self.harmonic_list:
            target_f = f_main * n
            idx = torch.argmin(torch.abs(_freq_array - target_f))
            f_h = float(_freq_array[idx].item())
            a_h = float(_mag_list[idx].item())
            self.main_harmonic.append((f_h,a_h))

        f_np = _freq_array.detach().cpu().numpy()
        m_np = _mag_list.detach().cpu().numpy()
        target = a_main / np.sqrt(2)
        f_low = None
        for k in range(main_idx-1, 0, -1):
            if m_np[k] >= target and m_np[k-1] < target:
                f1,f2 = f_np[k-1], f_np[k]
                m1,m2 = m_np[k-1], m_np[k]
                frac = (target - m1) / (m2 - m1 + 1e-30)
                f_low = f1 + frac*(f2 - f1)
                break

        f_high = None
        for k in range(main_idx+1, len(m_np)-1):
            if m_np[k] >= target and m_np[k+1] < target:
                f1,f2 = f_np[k], f_np[k+1]
                m1,m2 = m_np[k], m_np[k+1]
                frac = (target - m1) / (m2 - m1 + 1e-30)
                f_high = f1 + frac*(f2 - f1)
                break
        if (f_low is not None) and (f_high is not None) and (f_high > f_low):
            _BW = f_high - f_low
            _Q  = f_main / _BW if _BW > 0 else None
        else:
            raise ValueError("Unable to get BW")
        self.F0 = float(f_main)
        self.BW = float(_BW)
        self.Q = float(_Q)
        self.fft_result = _mag_list.cpu().numpy()
        
        return None
    def test_prt(self,_data):
        plt.figure(figsize=(8,4))
        plt.plot(_data)
        plt.xscale('log')  # 或 plt.xlim(0, fs/2)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dBV]")
        plt.title("FFT Spectrum of V(out)")
        plt.grid(True, which='both')
        plt.tight_layout()
        plt.show()
        return None
