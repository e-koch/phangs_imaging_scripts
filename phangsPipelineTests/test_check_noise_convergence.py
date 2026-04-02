"""
Unit tests for check_noise_convergence() in casaImagingRoutines.

These tests are fully self-contained: no CASA, no MS, no key files.
Mock tclean fullsummary dictionaries are constructed inline.

How to run:
    # Plain Python 3 (no CASA required):
    cd phangs_imaging_scripts
    python -m pytest phangsPipelineTests/test_check_noise_convergence.py -v

    # Inside CASA:
    sys.path.append('.')
    import phangsPipelineTests
    phangsPipelineTests.TestingCheckNoiseConvergenceInCasa().run()
"""

import sys
import unittest
import math
import numpy as np

# ---------------------------------------------------------------------------
# Import check_noise_convergence without triggering CASA-dependent imports.
# casaImagingRoutines imports analysisUtils and casaStuff at module level;
# we stub those out before importing so the module loads in plain Python 3.
# ---------------------------------------------------------------------------

def _import_check_noise_convergence():
    """
    Return check_noise_convergence with CASA stubs in place.

    casaImagingRoutines is loaded directly from its file via
    importlib.util.spec_from_file_location, bypassing phangsPipeline/__init__.py
    which would pull in the full package and its CASA-dependent imports.
    """
    import types
    import importlib.util
    import os

    # Stub every CASA/pipeline module that casaImagingRoutines imports at the
    # top level.  Only add stubs for names not already present (avoids clobbering
    # real modules when running inside CASA).
    stubs_needed = {
        'analysisUtils':                   {},
        'casaStuff':                       {},
        'pyfits':                          {},
        'phangsPipeline.casaStuff':        {},
        'phangsPipeline.casaMaskingRoutines': {},
        'phangsPipeline.pipelineVersion':  {'version': '0.0.0-test',
                                            'tableversion': '0'},
        'phangsPipeline.clean_call':       {'CleanCall': object},
    }
    for mod_name, attrs in stubs_needed.items():
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            for attr, val in attrs.items():
                setattr(stub, attr, val)
            sys.modules[mod_name] = stub

    # Load casaImagingRoutines directly from its file so phangsPipeline's
    # __init__.py (which imports the full package) is never executed.
    mod_name = 'phangsPipeline.casaImagingRoutines'
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
    else:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src  = os.path.join(here, 'phangsPipeline', 'casaImagingRoutines.py')
        spec = importlib.util.spec_from_file_location(mod_name, src)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

    return mod.check_noise_convergence


check_noise_convergence = _import_check_noise_convergence()


# ---------------------------------------------------------------------------
# Helper: build a minimal fullsummary dict
# ---------------------------------------------------------------------------

def _make_summary(cycles, field='0', chan='0', pol='0'):
    """
    Build a mock tclean fullsummary dict.

    Parameters
    ----------
    cycles : list of (iter_done, model_flux, start_model_flux)
        One tuple per major cycle.  Values are per-cycle scalars.
    field, chan, pol : str
        Keys used for the nested summaryminor structure.

    Returns
    -------
    dict  matching the structure of tclean(fullsummary=True)
    """
    iter_done    = np.array([c[0] for c in cycles])
    model_flux   = np.array([c[1] for c in cycles])
    start_flux   = np.array([c[2] for c in cycles])

    return {
        'summaryminor': {
            field: {
                chan: {
                    pol: {
                        'iterDone':       iter_done,
                        'modelFlux':      model_flux,
                        'startModelFlux': start_flux,
                    }
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckNoiseConvergence(unittest.TestCase):

    # ------------------------------------------------------------------
    # Guard-rail / edge cases
    # ------------------------------------------------------------------

    def test_none_summary_returns_none(self):
        self.assertIsNone(check_noise_convergence(None, noise=0.01, gain=0.1))

    def test_non_dict_summary_returns_none(self):
        self.assertIsNone(check_noise_convergence("not a dict", noise=0.01, gain=0.1))

    def test_missing_summaryminor_key_returns_none(self):
        self.assertIsNone(check_noise_convergence({}, noise=0.01, gain=0.1))

    def test_empty_summaryminor_returns_converged_zero(self):
        result = check_noise_convergence(
            {'summaryminor': {}}, noise=0.01, gain=0.1)
        self.assertIsNotNone(result)
        self.assertEqual(result['N_components'], 0)
        self.assertEqual(result['z_score'], 0.0)
        self.assertTrue(result['converged'])

    def test_zero_iter_done_returns_converged_zero(self):
        summary = _make_summary([(0, 0.0, 0.0)])
        result = check_noise_convergence(summary, noise=0.01, gain=0.1)
        self.assertIsNotNone(result)
        self.assertEqual(result['N_components'], 0)
        self.assertTrue(result['converged'])

    def test_missing_keys_in_data_gracefully_skipped(self):
        # Data dict is missing 'iterDone' — should return zero-component result
        summary = {
            'summaryminor': {
                '0': {'0': {'0': {'modelFlux': np.array([0.1])}}}
            }
        }
        result = check_noise_convergence(summary, noise=0.01, gain=0.1)
        self.assertIsNotNone(result)
        self.assertEqual(result['N_components'], 0)

    # ------------------------------------------------------------------
    # Noise-only case: z-score should be near zero
    # ------------------------------------------------------------------

    def test_noise_only_z_score_near_zero(self):
        """
        Simulate a tclean run on pure noise.  The flux increment per cycle
        is drawn from the expected noise distribution: each component adds
        ~gain*noise, so the total flux increment over N components should
        have |z| << 2.

        We use a deterministic case: N=100 components, flux_sum exactly 0
        (equal positive and negative noise components cancel).
        """
        noise, gain, N = 0.01, 0.1, 100
        summary = _make_summary([
            (N, 0.0, 0.0)  # flux_sum = modelFlux - startModelFlux = 0
        ])
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=1, z_threshold=2.0)

        self.assertIsNotNone(result)
        self.assertEqual(result['N_components'], N)
        self.assertAlmostEqual(result['flux_sum'], 0.0)
        self.assertAlmostEqual(
            result['std_expected'], math.sqrt(N) * gain * noise, places=10)
        self.assertAlmostEqual(result['z_score'], 0.0, places=10)
        self.assertTrue(result['converged'])

    def test_noise_only_small_positive_flux_still_converges(self):
        """
        A small flux increment (well within 1-sigma) should still converge.
        std_expected = sqrt(100) * 0.1 * 0.01 = 0.01
        flux_sum = 0.005  →  z = 0.5  →  converged
        """
        noise, gain, N = 0.01, 0.1, 100
        std = math.sqrt(N) * gain * noise   # 0.01
        flux_sum = 0.5 * std                # 0.005, z=0.5
        summary = _make_summary([(N, flux_sum, 0.0)])
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=1, z_threshold=2.0)

        self.assertAlmostEqual(result['z_score'], 0.5, places=8)
        self.assertTrue(result['converged'])

    # ------------------------------------------------------------------
    # Signal case: z-score should be large, not converged
    # ------------------------------------------------------------------

    def test_signal_present_z_score_large(self):
        """
        A systematic flux increment of 10*std should give z~10 and not converge.
        std_expected = sqrt(200) * 0.1 * 0.01 ≈ 0.01414
        flux_sum = 10 * std ≈ 0.1414
        """
        noise, gain, N = 0.01, 0.1, 200
        std = math.sqrt(N) * gain * noise
        flux_sum = 10.0 * std
        summary = _make_summary([(N, flux_sum, 0.0)])
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=1, z_threshold=2.0)

        self.assertAlmostEqual(result['z_score'], 10.0, places=6)
        self.assertFalse(result['converged'])

    def test_negative_signal_not_converged(self):
        """
        Negative systematic flux (|z| >> threshold) should not converge.
        """
        noise, gain, N = 0.01, 0.1, 100
        std = math.sqrt(N) * gain * noise
        flux_sum = -5.0 * std     # z = -5
        summary = _make_summary([(N, flux_sum, 0.0)])
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=1, z_threshold=2.0)

        self.assertAlmostEqual(result['z_score'], -5.0, places=6)
        self.assertFalse(result['converged'])

    # ------------------------------------------------------------------
    # Windowing: last_n_cycles
    # ------------------------------------------------------------------

    def test_window_uses_only_last_cycles(self):
        """
        First 3 cycles have large signal (z>>2), last 2 are noise-consistent.
        With last_n_cycles=2 the test should pass; with last_n_cycles=5 fail.
        """
        noise, gain = 0.01, 0.1

        # 5 cycles: first 3 have 5*std flux each, last 2 have 0 flux
        std_per_100 = math.sqrt(100) * gain * noise  # 0.01
        signal_flux = 5.0 * std_per_100

        cycles = [
            (100, signal_flux, 0.0),   # cycle 0 — signal
            (100, signal_flux, 0.0),   # cycle 1 — signal
            (100, signal_flux, 0.0),   # cycle 2 — signal
            (100, 0.0,         0.0),   # cycle 3 — noise
            (100, 0.0,         0.0),   # cycle 4 — noise
        ]
        summary = _make_summary(cycles)

        result_window2 = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=2, z_threshold=2.0)
        result_window5 = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=5, z_threshold=2.0)

        # Last 2 cycles: flux_sum=0, N=200 → z=0 → converged
        self.assertAlmostEqual(result_window2['z_score'], 0.0, places=8)
        self.assertTrue(result_window2['converged'])

        # All 5 cycles: flux_sum=3*signal_flux, N=500 → z>>2 → not converged
        self.assertGreater(abs(result_window5['z_score']), 2.0)
        self.assertFalse(result_window5['converged'])

    def test_window_larger_than_ncycles_uses_all(self):
        """
        Requesting more cycles than exist should use all available cycles
        without error.
        """
        noise, gain, N = 0.01, 0.1, 50
        summary = _make_summary([(N, 0.0, 0.0), (N, 0.0, 0.0)])
        # last_n_cycles=10 but only 2 cycles exist
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=10, z_threshold=2.0)
        self.assertEqual(result['N_components'], 2 * N)

    # ------------------------------------------------------------------
    # Multi-field / multi-channel accumulation
    # ------------------------------------------------------------------

    def test_multi_field_accumulation(self):
        """
        Two fields, each with 100 components and zero flux increment.
        Total N_components should be 200 and z_score should be 0.
        """
        noise, gain, N = 0.01, 0.1, 100
        summary = {
            'summaryminor': {
                '0': {'0': {'0': {
                    'iterDone':       np.array([N]),
                    'modelFlux':      np.array([0.0]),
                    'startModelFlux': np.array([0.0]),
                }}},
                '1': {'0': {'0': {
                    'iterDone':       np.array([N]),
                    'modelFlux':      np.array([0.0]),
                    'startModelFlux': np.array([0.0]),
                }}},
            }
        }
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=1, z_threshold=2.0)

        self.assertEqual(result['N_components'], 2 * N)
        self.assertAlmostEqual(result['z_score'], 0.0, places=10)

    def test_multi_channel_accumulation(self):
        """
        One field, two channels.  Channel 0 has signal (z>>2 alone),
        channel 1 has equal and opposite flux so that the combined z is 0.
        """
        noise, gain, N = 0.01, 0.1, 100
        std = math.sqrt(N) * gain * noise
        flux = 5.0 * std  # would give z=5 if alone

        summary = {
            'summaryminor': {
                '0': {
                    '0': {'0': {
                        'iterDone':       np.array([N]),
                        'modelFlux':      np.array([flux]),
                        'startModelFlux': np.array([0.0]),
                    }},
                    '1': {'0': {
                        'iterDone':       np.array([N]),
                        'modelFlux':      np.array([-flux]),
                        'startModelFlux': np.array([0.0]),
                    }},
                }
            }
        }
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=1, z_threshold=2.0)

        self.assertEqual(result['N_components'], 2 * N)
        self.assertAlmostEqual(result['flux_sum'], 0.0, places=10)
        self.assertAlmostEqual(result['z_score'], 0.0, places=8)

    # ------------------------------------------------------------------
    # z_threshold boundary
    # ------------------------------------------------------------------

    def test_z_exactly_at_threshold_not_converged(self):
        """
        |z| == threshold should NOT be declared converged (strict <).
        """
        noise, gain, N = 0.01, 0.1, 100
        std = math.sqrt(N) * gain * noise
        flux_sum = 2.0 * std   # z exactly = 2.0
        summary = _make_summary([(N, flux_sum, 0.0)])
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=1, z_threshold=2.0)

        self.assertAlmostEqual(result['z_score'], 2.0, places=8)
        self.assertFalse(result['converged'])   # not strictly < 2.0

    def test_z_just_below_threshold_converges(self):
        noise, gain, N = 0.01, 0.1, 100
        std = math.sqrt(N) * gain * noise
        flux_sum = 1.99 * std
        summary = _make_summary([(N, flux_sum, 0.0)])
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=1, z_threshold=2.0)

        self.assertLess(result['z_score'], 2.0)
        self.assertTrue(result['converged'])

    # ------------------------------------------------------------------
    # std_expected formula
    # ------------------------------------------------------------------

    def test_std_expected_formula(self):
        """
        std_expected = sqrt(N) * |gain| * noise
        Verify for several (N, gain, noise) combinations.
        """
        cases = [
            (100,  0.1,  0.01),
            (500,  0.1,  0.005),
            (50,   0.05, 0.02),
            (1000, 0.1,  0.001),
        ]
        for N, gain, noise in cases:
            summary = _make_summary([(N, 0.0, 0.0)])
            result = check_noise_convergence(
                summary, noise=noise, gain=gain, last_n_cycles=1)
            expected_std = math.sqrt(N) * abs(gain) * noise
            self.assertAlmostEqual(
                result['std_expected'], expected_std, places=12,
                msg='std_expected mismatch for N={}, gain={}, noise={}'.format(
                    N, gain, noise))

    # ------------------------------------------------------------------
    # n_pix / extreme-value correction
    # ------------------------------------------------------------------

    def test_extreme_value_std_formula(self):
        """
        With n_pix provided, std_expected = sqrt(N)*gain*noise*sqrt(2*ln(2*n_pix)).
        """
        N, gain, noise, n_pix = 100, 0.1, 0.01, 4096
        summary = _make_summary([(N, 0.0, 0.0)])
        result = check_noise_convergence(
            summary, noise=noise, gain=gain, last_n_cycles=1, n_pix=n_pix)
        expected_std = (math.sqrt(N) * gain * noise
                        * math.sqrt(2.0 * math.log(2.0 * n_pix)))
        self.assertAlmostEqual(result['std_expected'], expected_std, places=10)

    def test_extreme_value_noise_only_z_near_zero(self):
        """S=0 with n_pix → z=0 → converged."""
        summary = _make_summary([(100, 0.0, 0.0)])
        result = check_noise_convergence(
            summary, noise=0.01, gain=0.1,
            last_n_cycles=1, n_pix=4096, z_threshold=2.0)
        self.assertAlmostEqual(result['z_score'], 0.0, places=8)
        self.assertTrue(result['converged'])

    def test_extreme_value_signal_not_converged(self):
        """flux_sum = 10*std → z=10 → not converged."""
        N, gain, noise, n_pix = 100, 0.1, 0.01, 4096
        std = (math.sqrt(N) * gain * noise
               * math.sqrt(2.0 * math.log(2.0 * n_pix)))
        flux_sum = 10.0 * std
        summary = _make_summary([(N, flux_sum, 0.0)])
        result = check_noise_convergence(
            summary, noise=noise, gain=gain,
            last_n_cycles=1, n_pix=n_pix, z_threshold=2.0)
        self.assertAlmostEqual(result['z_score'], 10.0, places=6)
        self.assertFalse(result['converged'])

    def test_n_pix_none_uses_naive_fallback(self):
        """n_pix=None → old formula std = sqrt(N)*gain*noise."""
        N, gain, noise = 100, 0.1, 0.01
        summary = _make_summary([(N, 0.0, 0.0)])
        result = check_noise_convergence(
            summary, noise=noise, gain=gain, last_n_cycles=1, n_pix=None)
        self.assertAlmostEqual(
            result['std_expected'], math.sqrt(N) * gain * noise, places=10)

    def test_n_pix_1_uses_naive_fallback(self):
        """n_pix=1 (≤ 1 guard) → old formula."""
        summary = _make_summary([(100, 0.0, 0.0)])
        result = check_noise_convergence(
            summary, noise=0.01, gain=0.1, last_n_cycles=1, n_pix=1)
        self.assertAlmostEqual(
            result['std_expected'], math.sqrt(100) * 0.1 * 0.01, places=10)


# ---------------------------------------------------------------------------
# CASA wrapper
# ---------------------------------------------------------------------------

class TestingCheckNoiseConvergenceInCasa:
    """
    Wrapper to run TestingCheckNoiseConvergence inside CASA.

    Usage:
        import phangsPipelineTests
        phangsPipelineTests.TestingCheckNoiseConvergenceInCasa().run()
    """

    def __init__(self):
        pass

    def suite(self=None):
        testsuite = unittest.TestSuite()
        testsuite.addTest(unittest.makeSuite(TestCheckNoiseConvergence))
        return testsuite

    def run(self):
        unittest.main(
            defaultTest=(
                'phangsPipelineTests'
                '.TestingCheckNoiseConvergenceInCasa.suite'),
            verbosity=2,
            exit=False)


if __name__ == '__main__':
    unittest.main()
