import os
import sys
import types
import unittest
import numpy as np
import torch

# ----------------------------------------------------------------------
# Create minimal scipy stub so that processor module can be imported
scipy_stub = types.ModuleType('scipy')
signal_stub = types.ModuleType('signal')

# simple PSD using numpy's FFT
def simple_welch(x, fs=1.0, nperseg=None, noverlap=None, window=None):
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    psd = np.abs(np.fft.rfft(x))**2 / len(x)
    return freqs, psd

signal_stub.welch = simple_welch
signal_stub.savgol_filter = lambda x, window_length=0, polyorder=0: x
signal_stub.find_peaks = lambda x, height=None, prominence=None: (np.array([], dtype=int), {})
signal_stub.peak_widths = lambda *args, **kwargs: (np.array([0.0]), None, np.array([0.0]), np.array([0.0]))
signal_stub.iirnotch = lambda w0, Q: (np.array([1.0]), np.array([1.0]))
signal_stub.filtfilt = lambda b, a, x: x

scipy_stub.signal = signal_stub
io_stub = types.ModuleType('io')
io_stub.loadmat = lambda path: {}
scipy_stub.io = io_stub

sys.modules.setdefault('scipy', scipy_stub)
sys.modules.setdefault('scipy.signal', signal_stub)
sys.modules.setdefault('scipy.io', io_stub)

# ----------------------------------------------------------------------
# Add src directory to path and import module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from processor import WallPressureProcessor
import processing


class WallPressureProcessorTest(unittest.TestCase):
    def setUp(self):
        self.proc = WallPressureProcessor(
            sample_rate=1000,
            nu0=1e-5,
            rho0=1.0,
            u_tau0=0.1,
            err_frac=0.0,
            W=0.1,
            H=0.1,
            L0=1.0,
            delta_L0=0.0,
            U=10.0,
            C=20.0,
            mode_m=[0],
            mode_n=[0],
            mode_l=[0]
        )

    # --------------------------------------------------------------
    def test_compute_duct_modes(self):
        expected = processing.compute_duct_modes(
            10.0, 20.0, [0], [0], [0], 0.1, 0.1, 1.0, 1e-5, 0.1, 0.0
        )
        result = self.proc.compute_duct_modes()
        for k in expected:
            self.assertTrue(torch.equal(result[k], expected[k]))
        for k in expected:
            self.assertTrue(torch.equal(self.proc.duct_modes[k], expected[k]))

    # --------------------------------------------------------------
    def test_load_data(self):
        fs1 = torch.tensor([1000.0])
        p1 = torch.arange(5.0)
        fs2 = torch.tensor([1000.0])
        p2 = torch.arange(5.0) + 1
        def fake_loader(path):
            if 'wall' in path:
                return fs1, p1
            return fs2, p2
        module_path = 'processor.load_stan_wallpressure'
        with unittest.mock.patch(module_path, side_effect=fake_loader):
            res = self.proc.load_data('wall.mat', 'fs.mat')
        self.assertTrue(torch.equal(self.proc.f_w, fs1))
        self.assertTrue(torch.equal(self.proc.p_w, p1))
        self.assertEqual(res[0][0][0].item(), fs1[0].item())
        self.assertEqual(res[1][0][0].item(), fs2[0].item())

    # --------------------------------------------------------------
    def test_notch_filter(self):
        self.proc.p_w = torch.ones(10)
        self.proc.p_fs = torch.ones(10)
        fake_modes = {'nom':[1.0], 'min':[0.8], 'max':[1.2]}
        fake_res = (
            torch.zeros(10),
            torch.tensor([0.0,1.0]),
            torch.tensor([1.0,1.0]),
            torch.tensor([0.0,1.0]),
            torch.tensor([0.5,0.5]),
            [{'mode_freq':1.0}]
        )
        with unittest.mock.patch('processor.compute_duct_modes', return_value=fake_modes), \
             unittest.mock.patch('processor.notch_filter_timeseries', return_value=fake_res):
            self.proc.notch_filter()
        self.assertTrue(self.proc.filtered)
        self.assertTrue(hasattr(self.proc, 'p_w_filt'))
        self.assertEqual(len(self.proc.info_wall), 1)

    # --------------------------------------------------------------
    def test_compute_wall_spectrum(self):
        self.proc.p_w = torch.arange(10.0)
        with unittest.mock.patch('processor.compute_psd', return_value=(torch.tensor([0.0,1.0]), torch.tensor([1.0,2.0]))) as psd_mock:
            f, p = self.proc.compute_wall_spectrum()
        psd_mock.assert_called_once()
        self.assertEqual(list(f.numpy()), [0.0,1.0])
        self.assertEqual(list(p.numpy()), [1.0,2.0])

    # --------------------------------------------------------------
    def test_compute_transfer_function(self):
        self.proc.p_w = torch.arange(10.0)
        with unittest.mock.patch('processor.compute_psd', return_value=(torch.tensor([1.0,2.0]), torch.tensor([1.0,1.0]))):
            with unittest.mock.patch('processor.savgol_filter', side_effect=lambda x, window_length=0, polyorder=0: x):
                ref_f = np.array([1.0,2.0])
                ref_p = np.array([2.0,2.0])
                f, H = self.proc.compute_transfer_function(ref_f, ref_p)
        self.assertTrue(hasattr(self.proc, 'transfer_mag'))
        self.assertEqual(len(f), len(H))

    # --------------------------------------------------------------
    def test_apply_transfer_function(self):
        self.proc.p_w = torch.ones(8)
        self.proc.transfer_freq = torch.tensor([0.0, 500.0])
        self.proc.transfer_mag = torch.tensor([1.0, 0.5])
        corrected = self.proc.apply_transfer_function()
        self.assertEqual(len(corrected), len(self.proc.p_w))

    # --------------------------------------------------------------
    def test_phase_match(self):
        self.proc.p_w = torch.ones(4)
        self.proc.p_fs = torch.ones(4)
        self.proc.phase_match(smoothing_len=1)
        self.assertTrue(hasattr(self.proc, 'p_w_matched'))

    # --------------------------------------------------------------
    def test_reject_free_stream_noise(self):
        self.proc.p_w_matched = torch.zeros(4)
        self.proc.p_fs = torch.zeros(4)
        cleaned, f, H = self.proc.reject_free_stream_noise()
        self.assertEqual(len(cleaned), 4)


if __name__ == '__main__':
    unittest.main()
