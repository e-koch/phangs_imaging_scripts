"""
Write a complete set of PHANGS pipeline key files for a single target,
configuration, and spectral product.

Mirrors the full directory and key-file structure of the production
phangs-alma_keys setup (see phangs-alma_keys/master_key.txt), so that
KeyHandler reads without warnings about missing keys or directories.
Optional key files (singledish, cleanmask, distance, window, moment,
derived, linmos, override, dir) are written as comment-only stubs.

Pure Python — no CASA required.

Usage:
    from phangsPipelineTests.minimal_key_writer import MinimalKeyWriter
    writer = MinimalKeyWriter(
        key_dir='/path/to/test_keys/',
        data_dir='/path/to/test_data/',
        ms_root='/path/to/test_data/uvdata/',
    )
    master_key_path = writer.write_all(
        target='noise_only',
        config='C1',
        product='co21',
        ms_filename='noise_only.ms',
        restfreq_GHz=230.538,
        ra='12h00m00.0s',
        dec='-30d00m00.0s',
    )
"""

import os


class MinimalKeyWriter:
    """
    Write a complete PHANGS key set for a single target + interferometric
    config + line product.

    Directory layout under data_dir (all created on write_all()):
        imaging/
        postprocess/
        derived/
        release/
        singledish/
        cleanmasks/
        vfields/

    Key files written into key_dir:
        master_key.txt              — root configuration
        target_definitions.txt      — target positions / velocities
        config_definitions.txt      — array configs and spectral products
        ms_file_key.txt             — MS locations
        imaging_recipes.txt         — tclean recipe assignments
        noise_only.clean            — tclean parameter file for test imaging
        singledish_key.txt          — stub (no single-dish data)
        cleanmask_key.txt           — stub (no clean masks)
        distance_key.txt            — stub (no distance data)
        window_key.txt              — stub (no velocity windows)
        moment_key.txt              — stub (no moment definitions)
        derived_key.txt             — stub (no derived products)
        linearmosaic_definitions.txt — stub (no linear mosaics)
        overrides.txt               — stub (no parameter overrides)
        dir_key.txt                 — stub (no directory remapping)

    Parameters
    ----------
    key_dir : str
        Directory where key files will be written.
    data_dir : str
        Root directory for all pipeline output trees.
    ms_root : str
        Directory containing the measurement set file.
    """

    # Sub-directories of data_dir that mirror the production layout
    _DATA_SUBDIRS = [
        'uvdata',
        'imaging',
        'postprocess',
        'derived',
        'release',
        'singledish',
        'cleanmasks',
        'vfields',
    ]

    def __init__(self, key_dir, data_dir, ms_root):
        self.key_dir  = os.path.abspath(key_dir)
        self.data_dir = os.path.abspath(data_dir)
        self.ms_root  = os.path.abspath(ms_root)

    def write_all(self, target, config, product, ms_filename,
                  restfreq_GHz, ra, dec):
        """
        Write all key files and return the path to master_key.txt.

        Parameters
        ----------
        target : str
            Target name used consistently across all key files.
        config : str
            Interferometric configuration name (e.g. 'C1').
        product : str
            Spectral line product name (e.g. 'co21').
        ms_filename : str
            Filename of the MS relative to ms_root (e.g. 'noise_only.ms').
        restfreq_GHz : float
            Rest frequency of the line in GHz.
        ra : str
            Phase centre RA string (e.g. '12h00m00.0s').
        dec : str
            Phase centre Dec string (e.g. '-30d00m00.0s').

        Returns
        -------
        str
            Absolute path to master_key.txt.
        """
        os.makedirs(self.key_dir, exist_ok=True)
        for subdir in self._DATA_SUBDIRS:
            os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)

        self._write_master_key()
        self._write_target_definitions(target, ra, dec)
        self._write_config_definitions(config, product)
        self._write_ms_file_key(target, config, ms_filename)
        self._write_imaging_recipes(product)
        self._write_noise_only_clean()

        # Optional stub key files — empty but present so KeyHandler finds them
        self._write_stub('singledish_key.txt',
                         'target  product  singledish_fits_path')
        self._write_stub('cleanmask_key.txt',
                         'target  product  cleanmask_fits_path')
        self._write_stub('distance_key.txt',
                         'target,dist')
        self._write_stub('window_key.txt',
                         'target,window,vfield_file')
        self._write_stub('moment_key.txt',
                         'moment_name  keyword  value')
        self._write_stub('derived_key.txt',
                         'config  product  keyword  value')
        self._write_stub('linearmosaic_definitions.txt',
                         'mosaic_target  member_target')
        self._write_stub('overrides.txt',
                         'imagename  parameter  value')
        self._write_stub('dir_key.txt',
                         'pointing_target  galaxy_target')

        return os.path.join(self.key_dir, 'master_key.txt')

    # ------------------------------------------------------------------
    # Private writers
    # ------------------------------------------------------------------

    def _write_master_key(self):
        d = self.data_dir
        lines = [
            '# Auto-generated master key for pipeline tests\n',
            '# Mirrors the structure of phangs-alma_keys/master_key.txt\n',
            '\n',
            'key_dir             {}/\n'.format(self.key_dir),
            '\n',
            'ms_root             {}/\n'.format(
                os.path.join(d, 'uvdata')),
            'imaging_root        {}/\n'.format(
                os.path.join(d, 'imaging')),
            'postprocess_root    {}/\n'.format(
                os.path.join(d, 'postprocess')),
            'derived_root        {}/\n'.format(
                os.path.join(d, 'derived')),
            'release_root        {}/\n'.format(
                os.path.join(d, 'release')),
            'singledish_root     {}/\n'.format(
                os.path.join(d, 'singledish')),
            'cleanmask_root      {}/\n'.format(
                os.path.join(d, 'cleanmasks')),
            'vfield_root         {}/\n'.format(
                os.path.join(d, 'vfields')),
            '\n',
            'ms_key              ms_file_key.txt\n',
            'config_key          config_definitions.txt\n',
            'target_key          target_definitions.txt\n',
            'imaging_key         imaging_recipes.txt\n',
            'singledish_key      singledish_key.txt\n',
            'cleanmask_key       cleanmask_key.txt\n',
            'distance_key        distance_key.txt\n',
            'window_key          window_key.txt\n',
            'moment_key          moment_key.txt\n',
            'derived_key         derived_key.txt\n',
            'linmos_key          linearmosaic_definitions.txt\n',
            'override_key        overrides.txt\n',
            'dir_key             dir_key.txt\n',
        ]
        self._write_file('master_key.txt', lines)

    def _write_target_definitions(self, target, ra, dec):
        # velocity_kms=0 (vsys), width_kms=15:
        # 15 km/s ≈ 11.5 MHz at 230.538 GHz, which fits within the 18 MHz
        # simulated bandwidth (12 channels × 1.5 MHz) and selects ~8 channels
        # after SPW staging — satisfying the >4-channel requirement.
        lines = [
            '# Auto-generated target definitions for pipeline tests\n',
            '# Columns: target  ra  dec  velocity_kms  width_kms\n',
            '{}\t{}\t{}\t0\t15\n'.format(target, ra, dec),
        ]
        self._write_file('target_definitions.txt', lines)

    def _write_config_definitions(self, config, product):
        # Both 'array_tags' and 'clean_scales_arcsec' must be present:
        # print_configs() accesses them directly without a try/except.
        # The array_tag must also match the array_tag column in ms_file_key.txt.
        #
        # line_product requires 'line_tag' (raises if missing in task_split)
        # and 'channel_kms' (raises if missing in task_split).
        # 'restfreq' is not a valid line_product field — the rest frequency
        # is looked up from utilsLines via line_tag.
        lines = [
            '# Auto-generated config definitions for pipeline tests\n',
            '# Columns: type  name  parameters\n',
            "interf_config\t{config}\t{{'array_tags':['{config}'],"
            "'clean_scales_arcsec':[0]}}\n".format(config=config),
            "line_product\t{product}\t{{'line_tag':'{product}',"
            "'channel_kms':2.6}}\n".format(product=product),
        ]
        self._write_file('config_definitions.txt', lines)

    def _write_ms_file_key(self, target, config, ms_filename):
        lines = [
            '# Auto-generated MS file key for pipeline tests\n',
            '# Columns: target  project  field  array_tag  obs_num  filepath\n',
            '{}\ttest_proj\tall\t{}\t1\t{}\n'.format(
                target, config, ms_filename),
        ]
        self._write_file('ms_file_key.txt', lines)

    def _write_imaging_recipes(self, product):
        lines = [
            '# Auto-generated imaging recipes for pipeline tests\n',
            '# Columns: config  product  stage  recipe_file\n',
            'all\t{}\tall\tnoise_only.clean\n'.format(product),
        ]
        self._write_file('imaging_recipes.txt', lines)

    def _write_stub(self, filename, column_header):
        """Write a header-only (no data) stub key file."""
        lines = [
            '# Auto-generated stub for pipeline tests — no entries\n',
            '# Columns: {}\n'.format(column_header),
        ]
        self._write_file(filename, lines)

    def _write_noise_only_clean(self):
        """
        Write a minimal tclean parameter file for a noise-only,
        single-pointing MS produced by simobserve.

        Key differences from the production cube_mosaic.clean:
          - datacolumn  = "data"      (simobserve creates DATA, not CORRECTED)
          - gridder     = "standard"  (single pointing, no mosaic)
          - usemask     = "pb"        (no external clean mask needed)
          - imsize      = [64]        (small for fast tests)
          - cell        = ['1.0arcsec']
          - scales      = []          (point source only for speed)
        """
        lines = [
            '# Auto-generated tclean parameter file for noise-only tests\n',
            'vis                       =  ""\n',
            'selectdata                =  True\n',
            'field                     =  ""\n',
            'spw                       =  ""\n',
            'timerange                 =  ""\n',
            'uvrange                   =  ""\n',
            'antenna                   =  ""\n',
            'scan                      =  ""\n',
            'observation               =  ""\n',
            'intent                    =  ""\n',
            # simobserve creates DATA column, not CORRECTED_DATA
            'datacolumn                =  "data"\n',
            'imagename                 =  ""\n',
            'imsize                    =  [64]\n',
            "cell                      =  ['1.0arcsec']\n",
            'phasecenter               =  ""\n',
            'stokes                    =  "I"\n',
            'projection                =  "SIN"\n',
            'startmodel                =  ""\n',
            'specmode                  =  "cube"\n',
            'reffreq                   =  ""\n',
            'nchan                     =  -1\n',
            'start                     =  ""\n',
            'width                     =  ""\n',
            'outframe                  =  "LSRK"\n',
            'veltype                   =  "radio"\n',
            'restfreq                  =  []\n',
            'interpolation             =  "linear"\n',
            'perchanweightdensity      =  True\n',
            # Single pointing — standard gridder, not mosaic
            'gridder                   =  "standard"\n',
            'facets                    =  1\n',
            'psfphasecenter            =  ""\n',
            'chanchunks                =  1\n',
            'wprojplanes               =  1\n',
            'vptable                   =  ""\n',
            'pblimit                   =  0.2\n',
            'normtype                  =  "flatnoise"\n',
            'deconvolver               =  "multiscale"\n',
            'scales                    =  []\n',
            'nterms                    =  1\n',
            'smallscalebias            =  0.9\n',
            'restoration               =  True\n',
            'restoringbeam             =  "common"\n',
            'pbcor                     =  False\n',
            'outlierfile               =  ""\n',
            'weighting                 =  "briggs"\n',
            'robust                    =  0.5\n',
            "noise                     =  '1.0Jy'\n",
            'npixels                   =  0\n',
            'uvtaper                   =  []\n',
            'niter                     =  1\n',
            'gain                      =  0.1\n',
            "threshold                 =  '0.0mJy/beam'\n",
            'nsigma                    =  0.0\n',
            'cycleniter                =  200\n',
            'cyclefactor               =  3.0\n',
            'minpsffraction            =  0.5\n',
            'maxpsffraction            =  0.8\n',
            'interactive               =  False\n',
            # Use PB mask only — no external clean mask file required
            'usemask                   =  "pb"\n',
            'mask                      =  ""\n',
            'pbmask                    =  0.2\n',
            'sidelobethreshold         =  3.0\n',
            'noisethreshold            =  5.0\n',
            'lownoisethreshold         =  1.5\n',
            'negativethreshold         =  0.0\n',
            'smoothfactor              =  1.0\n',
            'minbeamfrac               =  0.3\n',
            'cutthreshold              =  0.01\n',
            'growiterations            =  75\n',
            'dogrowprune               =  True\n',
            'minpercentchange          =  -1.0\n',
            'verbose                   =  False\n',
            'fastnoise                 =  True\n',
            'restart                   =  True\n',
            'savemodel                 =  "none"\n',
            'calcres                   =  True\n',
            'calcpsf                   =  True\n',
            'parallel                  =  False\n',
        ]
        self._write_file('noise_only.clean', lines)

    def _write_file(self, filename, lines):
        filepath = os.path.join(self.key_dir, filename)
        with open(filepath, 'w') as f:
            f.writelines(lines)
