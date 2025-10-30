import numpy as np
import torch,threading,time,multiprocessing
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
import threading

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class calculator:
    _lock = threading.Lock()
    _task_completed = False
    harmonic_list = [1,2,3,4,5,6,7,8,9,10,11]
    unreach_time = 0.05
    Vpp_min = 0.01
    def __init__(self,_time_array,_y_array):
        while True:
            with self._lock:
                if not self._task_completed:
                    self.x = _time_array
                    self.y = _y_array
                    self.resonate_detector()
                    self.fft_result = None
                    self.stop_resonate_time = 1e-4
                    if self.stop_resonate_time is not None:
                        #self.resonate_only(1e-4)
                        # print(f"{len(self.y) = }")
                        # plt.figure(figsize=(12,4))
                        # plt.plot(self.x, self.y, lw=0.8)
                        # plt.xlabel("Time [s]")
                        # plt.ylabel("Amplitude [V]")
                        # plt.title("Time Domain Waveform")
                        # plt.grid(True, ls='--', alpha=0.7)
                        # plt.tight_layout()

                        # plt.savefig("./waveform_time.png", dpi=200)
                        # plt.close()
                        self.FFT()
                    self._task_completed = False
                    break
                else:
                    time.sleep(1)
                    continue
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
        i=0
        _Vpp_list = []
        for i in range(2, peak_idx.numel()-1):
            a1 = _z[peak_idx[i]].item()
            a2 = _z[peak_idx[i+1]].item()
            _Vpp_list.append((a1-a2,_x[peak_idx[i]].item()))
            
            # if a1 >= thr and a2 < thr:
            #     t1 = _x[peak_idx[i]].item()
            #     t2 = _x[peak_idx[i+1]].item()
            #     frac = (thr - a1) / (a2 - a1 + 1e-30)
            #     t_stop = t1 + frac * (t2 - t1)
            #     self.stop_resonate_time = float(t_stop)
            #     break
        for _Vpp in _Vpp_list:
            if abs(_Vpp[0]) < thr:
                t_stop = _Vpp[1]
                self.stop_resonate_time = float(t_stop)
                break
        if self.stop_resonate_time is None and peak_idx.numel() < 500:
            t_stop = float(_x[peak_idx[-1]].item())
            self.stop_resonate_time = float(t_stop)

        if self.stop_resonate_time is None:
            self.stop_resonate_time = self.unreach_time
        if self.stop_resonate_time > self.unreach_time:
            self.stop_resonate_time = self.unreach_time
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
        def lomb_batched_fft_gpu_core():
            
            _min_freq = 0.1/self.stop_resonate_time
            if _min_freq < 1e5:
                _min_freq = 1e5
            _max_freq = _min_freq*1e5
            if _max_freq > 1e8:
                _max_freq = 1e8
            _ctype = torch.complex64
            _x = torch.from_numpy(self.x).to(DEVICE).to(torch.float64)
            _y_c = torch.from_numpy(self.y - self.y.mean()).to(DEVICE).to(_ctype)
            _win = torch.hann_window(_x.numel(), periodic=True, device=DEVICE)
            _f = torch.linspace(float(_min_freq), float(_max_freq), int(1e7), device=DEVICE, dtype=torch.float64)
            _N = _x.numel()
            _Y = torch.zeros((_f.numel(),), device=DEVICE, dtype=_ctype)
            bytes_per_complex = 8
            target_bytes = 1024 * 1024**2
            est_chunk = max(1024, int(target_bytes / (_N * bytes_per_complex + 1)))
            batch_size = min(est_chunk, _f.numel())
            #batch_size = int(0.5e4)
            for i in range(0, _f.numel(), batch_size):
                f_chunk = _f[i:i+batch_size]                        # (C,)
                w = 2.0 * torch.pi * f_chunk[:, None] * _x[None, :]      # (C,N) float64
                exp_term = torch.exp((-1j * w).to(_ctype))                 # (C,N) complex
                _Y[i:i+batch_size] = torch.matmul(exp_term, _y_c) / _N            # (C,)
                del w, exp_term
            _mag = torch.abs(_Y)
            _mag = _mag * 2.0 / (_win.sum()/ _N) / _N
            del _x,_y_c,_win,_Y

            # f_np = _f.detach().cpu().numpy()
            # mag_np = _mag.detach().cpu().numpy()

            # plt.figure(figsize=(16,3))
            # plt.semilogx(f_np, mag_np) 

            # plt.xlabel("Frequency [Hz]")
            # plt.ylabel("Magnitude")
            # plt.title("Spectrum (Log–Log Scale)")
            # plt.grid(True, which="both", ls="--", alpha=0.7)

            # plt.tight_layout()
            # plt.savefig("./spectrum.png", dpi=200)
            # plt.close()
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
