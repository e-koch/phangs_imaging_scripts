"""
Write a minimal set of PHANGS pipeline key files for a single target,
configuration, and spectral product.

Pure Python — no CASA required. Intended for use in pipeline tests that need
a complete but lightweight key setup without referencing real data paths.

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
    Write the five key files required by KeyHandler to run imaging for a
    single target + interferometric config + line product.

    Files written into key_dir:
        master_key.txt
        target_definitions.txt
        config_definitions.txt
        ms_file_key.txt
        imaging_recipes.txt
        noise_only.clean       (tclean parameter file for test imaging)

    The imaging_root and ms_root directories are created if missing.

    Parameters
    ----------
    key_dir : str
        Directory where key files will be written.
    data_dir : str
        Root directory for imaging output (imaging_root = data_dir/imaging/).
    ms_root : str
        Directory containing the measurement set file.
    """

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
        imaging_root = os.path.join(self.data_dir, 'imaging')
        os.makedirs(imaging_root, exist_ok=True)
        os.makedirs(self.ms_root, exist_ok=True)

        self._write_master_key(imaging_root)
        self._write_target_definitions(target, ra, dec)
        self._write_config_definitions(config, product, restfreq_GHz)
        self._write_ms_file_key(target, config, ms_filename)
        self._write_imaging_recipes(product)
        self._write_noise_only_clean()

        return os.path.join(self.key_dir, 'master_key.txt')

    # ------------------------------------------------------------------
    # Private writers
    # ------------------------------------------------------------------

    def _write_master_key(self, imaging_root):
        lines = [
            '# Auto-generated master key for pipeline tests\n',
            'key_dir         {}/\n'.format(self.key_dir),
            'imaging_root    {}/\n'.format(imaging_root),
            'ms_root         {}/\n'.format(self.ms_root),
            'ms_key          ms_file_key.txt\n',
            'config_key      config_definitions.txt\n',
            'target_key      target_definitions.txt\n',
            'imaging_key     imaging_recipes.txt\n',
        ]
        self._write_file('master_key.txt', lines)

    def _write_target_definitions(self, target, ra, dec):
        lines = [
            '# Auto-generated target definitions for pipeline tests\n',
            # columns: target  ra  dec  velocity_kms  width_kms
            '{target}\t{ra}\t{dec}\t0\t200\n'.format(
                target=target, ra=ra, dec=dec),
        ]
        self._write_file('target_definitions.txt', lines)

    def _write_config_definitions(self, config, product, restfreq_GHz):
        lines = [
            '# Auto-generated config definitions for pipeline tests\n',
            'interf_config\t{config}\t{{}}\n'.format(config=config),
            "line_product\t{product}\t{{'restfreq':{restfreq}}}\n".format(
                product=product, restfreq=restfreq_GHz),
        ]
        self._write_file('config_definitions.txt', lines)

    def _write_ms_file_key(self, target, config, ms_filename):
        lines = [
            '# Auto-generated MS file key for pipeline tests\n',
            # columns: target  project  field  array_tag  obs_num  filepath
            '{target}\ttest_proj\tall\t{config}\t1\t{msfile}\n'.format(
                target=target, config=config, msfile=ms_filename),
        ]
        self._write_file('ms_file_key.txt', lines)

    def _write_imaging_recipes(self, product):
        lines = [
            '# Auto-generated imaging recipes for pipeline tests\n',
            # columns: config  product  stage  recipe_file
            'all\t{product}\tall\tnoise_only.clean\n'.format(product=product),
        ]
        self._write_file('imaging_recipes.txt', lines)

    def _write_noise_only_clean(self):
        """
        Write a minimal tclean parameter file suitable for a noise-only,
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
