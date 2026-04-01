"""
Integration test for the noise-convergence stopping criterion in clean_loop().

Tests that check_noise_convergence() correctly identifies imaging runs that
are cleaning pure noise (z-score near zero) versus runs where the statistic
is disabled (z-score column shows N/A).

How to run this test inside CASA:

    sys.path.append('../analysis_scripts')
    sys.path.append('.')
    import phangsPipeline
    import phangsPipelineTests
    phangsPipelineTests.TestingNoiseConvergenceInCasa().run()

What is tested:

    1. test_noise_z_score_computed_and_low
       Runs multiscale clean on a noise-only MS with
       convergence_noise_z_threshold=2.0 and convergence_fracflux=None.
       Asserts that the noise_z_score column in the record file contains
       numeric values and that the final z-score is < 2.0.

    2. test_noise_z_score_disabled
       Same setup but convergence_noise_z_threshold=None.
       Asserts that every noise_z_score entry in the record file is 'N/A'.

Setup strategy:

    setUpClass() runs once for the whole test class:
      - creates the noise-only MS via simobserve
      - writes minimal pipeline key files
      - runs loop_stage_uvdata (HandlerVis) to stage the MS into the imaging tree
      - makes one dirty image so each test can revert to it cheaply

    Each test method only calls loop_imaging with
      do_dirty_image=False, do_revert_to_dirty=True, do_multiscale_clean=True
    so the expensive simobserve and staging steps are not repeated.
"""

import os
import sys
import glob
import unittest

from .noise_only_ms_factory import NoiseOnlyMSFactory
from .minimal_key_writer import MinimalKeyWriter

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

_TARGET  = 'noise_only'
_CONFIG  = 'C1'
_PRODUCT = 'co21'
_RA      = '12h00m00.0s'
_DEC     = '-30d00m00.0s'
_RESTFREQ_GHZ = 230.538

# ---------------------------------------------------------------------------
# Record file parsing helpers
# ---------------------------------------------------------------------------

def _read_record_file(record_file):
    """
    Return list of data rows from a clean_loop record file, each row as a
    list of stripped strings.  Comment lines (starting with #) are skipped.
    """
    rows = []
    with open(record_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([x.strip() for x in line.split(',')])
    return rows


def _parse_z_scores_raw(record_file):
    """
    Return list of raw noise_z_score strings from the record file.

    The first non-comment line is the column header; subsequent lines are
    data.  The noise_z_score column is the last column.
    """
    rows = _read_record_file(record_file)
    if len(rows) < 2:
        return []
    # rows[0] is the header, rows[1:] are data rows
    return [row[-1] for row in rows[1:]]


def _parse_z_scores(record_file):
    """
    Return list of noise_z_score values as floats, skipping any 'N/A' entries.
    """
    raw = _parse_z_scores_raw(record_file)
    result = []
    for val in raw:
        try:
            result.append(float(val))
        except ValueError:
            pass
    return result


def _find_record_file(imaging_dir, stage='multiscale'):
    """
    Locate the clean_loop record file for the given stage under imaging_dir.
    Returns None if not found.
    """
    pattern = os.path.join(imaging_dir, '**',
                           '*_{}_record.txt'.format(stage))
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestingNoiseConvergence(unittest.TestCase):
    """
    Integration tests for the noise-convergence stopping criterion.

    setUpClass() runs once: creates the noise-only MS, writes key files,
    stages the UV data via HandlerVis, and makes a dirty image.

    Individual test methods revert to the dirty image and run multiscale
    clean with different convergence settings.
    """

    @classmethod
    def setUpClass(cls):
        """
        One-time setup for the entire test class:
          1. Create noise-only MS via simobserve.
          2. Write minimal pipeline key files.
          3. Stage the MS into the imaging tree via loop_stage_uvdata.
          4. Make one dirty image so tests can cheaply revert to it.
        """
        import phangsPipeline
        from phangsPipeline import handlerKeys as kh
        from phangsPipeline import handlerVis as uvh
        from phangsPipeline import handlerImaging as imh

        cls.current_dir  = os.getcwd()
        cls.module_dir   = os.path.dirname(
            os.path.abspath(phangsPipeline.__path__[0]))
        cls.working_dir  = os.path.join(cls.module_dir, 'phangsPipelineTests')
        cls.test_data_dir = os.path.join(cls.working_dir,
                                         'test_data', 'noise_only')
        cls.test_keys_dir = os.path.join(cls.working_dir,
                                         'test_keys', 'noise_only')
        cls.ms_root      = os.path.join(cls.test_data_dir, 'uvdata')
        cls.imaging_dir  = os.path.join(cls.test_data_dir, 'imaging')

        os.chdir(cls.working_dir)

        # Wipe any artefacts left by a previous run so each run starts clean.
        import shutil
        for d in [cls.test_data_dir, cls.test_keys_dir]:
            if os.path.isdir(d):
                shutil.rmtree(d)

        for d in [cls.test_data_dir, cls.test_keys_dir,
                  cls.ms_root, cls.imaging_dir]:
            os.makedirs(d, exist_ok=True)

        # 1. Build the noise-only MS
        factory = NoiseOnlyMSFactory()
        cls.ms_path = factory.create(cls.ms_root)

        # 2. Write pipeline key files
        writer = MinimalKeyWriter(
            key_dir=cls.test_keys_dir,
            data_dir=cls.test_data_dir,
            ms_root=cls.ms_root,
        )
        cls.master_key = writer.write_all(
            target=_TARGET,
            config=_CONFIG,
            product=_PRODUCT,
            ms_filename='noise_only.ms',
            restfreq_GHz=_RESTFREQ_GHZ,
            ra=_RA,
            dec=_DEC,
        )

        # 3. Stage the UV data via HandlerVis.
        #    The simulated MS uses 1.5 MHz channels (< max_chanwidth ≈ 2 MHz
        #    from channel_kms=2.6) and 12 channels (18 MHz total bandwidth),
        #    which covers the 15 km/s target velocity window.
        os.chdir(cls.working_dir)
        this_kh = kh.KeyHandler(master_key=cls.master_key)
        this_uvh = uvh.VisHandler(key_handler=this_kh)
        this_uvh.set_targets(only=[_TARGET])
        this_uvh.set_interf_configs(only=[_CONFIG])
        this_uvh.set_line_products(only=[_PRODUCT])
        this_uvh.loop_stage_uvdata(do_all=True)

    def setUp(self):
        os.chdir(self.working_dir)

    def tearDown(self):
        os.chdir(self.current_dir)

    # Optional post-run cleanup — disabled by default so generated files
    # can be inspected after a run.  Uncomment to clean up automatically.
    @classmethod
    def tearDownClass(cls):
        import shutil
        os.chdir(cls.current_dir)
        for d in [cls.test_data_dir, cls.test_keys_dir]:
            if os.path.isdir(d):
                shutil.rmtree(d)

    # ------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------

    def test_noise_z_score_computed_and_low(self):
        """
        Run multiscale clean on noise-only data with the noise convergence
        criterion enabled.  Verify that:
          - the noise_z_score column contains numeric values (not N/A), and
          - the final z-score is < 2.0, consistent with noise-only components.
        """
        from phangsPipeline import handlerKeys as kh
        from phangsPipeline import handlerImaging as imh

        this_kh = kh.KeyHandler(master_key=self.master_key)
        this_imh = imh.ImagingHandler(key_handler=this_kh)
        this_imh.set_targets(only=[_TARGET])
        this_imh.set_interf_configs(only=[_CONFIG])
        this_imh.set_line_products(only=[_PRODUCT])

        this_imh.loop_imaging(
            do_dirty_image=True,
            do_revert_to_dirty=True,
            do_read_clean_mask=False,
            do_multiscale_clean=True,
            do_revert_to_multiscale=False,
            do_singlescale_mask=False,
            do_singlescale_clean=False,
            do_revert_to_singlescale=False,
            do_export_to_fits=False,
            # Disable flux-change criterion so the noise check is the
            # primary convergence mechanism
            convergence_fracflux=None,
            convergence_noise_z_threshold=5.0,
            overwrite=True,
        )

        record_file = _find_record_file(self.imaging_dir, stage='multiscale')
        self.assertIsNotNone(
            record_file,
            'Could not find multiscale_record.txt under ' + self.imaging_dir)

        z_scores = _parse_z_scores(record_file)
        self.assertGreater(
            len(z_scores), 0,
            'No numeric z-scores found in record file: ' + record_file)

        final_z = z_scores[-1]
        self.assertLess(
            final_z, 2.0,
            'Final noise z-score {:.3f} >= 2.0 for noise-only data'.format(
                final_z))

    def test_noise_z_score_disabled(self):
        """
        Run multiscale clean with convergence_noise_z_threshold=None.
        Verify that every noise_z_score entry in the record file is 'N/A'.
        """
        from phangsPipeline import handlerKeys as kh
        from phangsPipeline import handlerImaging as imh

        this_kh = kh.KeyHandler(master_key=self.master_key)
        this_imh = imh.ImagingHandler(key_handler=this_kh)
        this_imh.set_targets(only=[_TARGET])
        this_imh.set_interf_configs(only=[_CONFIG])
        this_imh.set_line_products(only=[_PRODUCT])

        this_imh.loop_imaging(
            do_dirty_image=False,
            do_revert_to_dirty=True,
            do_read_clean_mask=False,
            do_multiscale_clean=True,
            do_revert_to_multiscale=False,
            do_singlescale_mask=False,
            do_singlescale_clean=False,
            do_revert_to_singlescale=False,
            do_export_to_fits=False,
            convergence_fracflux=0.01,
            convergence_noise_z_threshold=None,
            overwrite=True,
        )

        record_file = _find_record_file(self.imaging_dir, stage='multiscale')
        self.assertIsNotNone(
            record_file,
            'Could not find multiscale_record.txt under ' + self.imaging_dir)

        raw_z = _parse_z_scores_raw(record_file)
        self.assertGreater(
            len(raw_z), 0,
            'No z-score entries found in record file: ' + record_file)

        non_na = [v for v in raw_z if v != 'N/A']
        self.assertEqual(
            len(non_na), 0,
            'Expected all N/A when noise threshold disabled, '
            'but found numeric values: ' + str(non_na))


# ---------------------------------------------------------------------------
# CASA wrapper (mirrors TestingHandlerImagingInCasa pattern)
# ---------------------------------------------------------------------------

class TestingNoiseConvergenceInCasa:
    """
    Wrapper to run TestingNoiseConvergence inside CASA's Python environment.

    Usage:
        import phangsPipelineTests
        phangsPipelineTests.TestingNoiseConvergenceInCasa().run()
    """

    def __init__(self):
        pass

    def suite(self=None):
        testsuite = unittest.TestSuite()
        testsuite.addTest(unittest.makeSuite(TestingNoiseConvergence))
        return testsuite

    def run(self):
        unittest.main(
            defaultTest='phangsPipelineTests.TestingNoiseConvergenceInCasa.suite',
            verbosity=2,
            exit=False)


if __name__ == '__main__':
    unittest.main()
