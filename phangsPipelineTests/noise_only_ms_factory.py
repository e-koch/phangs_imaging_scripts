"""
Factory for creating a noise-only ALMA measurement set via CASA's simobserve.

Intended for use in pipeline tests that need a realistic but signal-free MS.
All CASA imports are deferred to method bodies so this module can be imported
in a plain Python 3 environment without CASA present.

Usage (inside CASA):
    from phangsPipelineTests.noise_only_ms_factory import NoiseOnlyMSFactory
    ms_path = NoiseOnlyMSFactory().create('/path/to/uvdata_dir/')
    # ... run tests ...
    NoiseOnlyMSFactory().cleanup('/path/to/uvdata_dir/')
"""

import os
import shutil

# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # Sky / pointing
    'target_name':        'noise_only',
    'ra':                 '12h00m00.0s',
    'dec':                '-30d00m00.0s',

    # Spectral setup (CO 2-1 as representative line)
    'restfreq_GHz':       230.538,
    'chan_width_MHz':     5.0,
    'nchan':              8,

    # Observing
    'integration_s':      30.0,    # per integration
    'total_time_s':       300.0,   # total on-source time

    # Noise: added per visibility per baseline per integration
    # 'noise_jy':           0.05,

    # ALMA antenna configuration bundled with CASA 6.4+.
    # C-1 (most compact) gives fast simulations with reasonable UV coverage.
    # Switch to 'alma.cycle10.2.cfg' for C-2 if broader UV coverage is needed.
    'antennalist':        'alma.cycle10.1.cfg',

    # Sky model image (blank, all zeros)
    'image_size_pix':     32,
    'pixel_size_arcsec':  1.0,
}

# Name used for the simobserve project subdirectory and the final MS file
_PROJECT_NAME = 'noise_sim'
_MS_FILENAME  = 'noise_only.ms'


class NoiseOnlyMSFactory:
    """
    Create and optionally clean up a noise-only ALMA MS for pipeline tests.

    The MS is produced by simobserve with a blank (all-zero) sky model and
    simplenoise thermal noise. The output DATA column (not CORRECTED_DATA)
    contains purely thermal noise — there is no astrophysical signal.

    Parameters
    ----------
    None — all simulation parameters are passed to create().
    """

    def create(self, output_dir, params=None):
        """
        Generate a noise-only MS and place it at output_dir/noise_only.ms.

        Parameters
        ----------
        output_dir : str
            Directory where noise_only.ms will be written. Created if absent.
        params : dict, optional
            Override any keys in DEFAULT_PARAMS.

        Returns
        -------
        str
            Absolute path to the created MS (output_dir/noise_only.ms).
        """
        # Merge parameter overrides
        p = dict(DEFAULT_PARAMS)
        if params is not None:
            p.update(params)

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        orig_dir = os.getcwd()

        try:
            # simobserve writes everything relative to cwd
            os.chdir(output_dir)

            sky_fits = os.path.join(output_dir, 'blank_sky.fits')
            self._write_blank_sky(sky_fits, p)

            self._run_simobserve(sky_fits, p)

            ms_dst = os.path.join(output_dir, _MS_FILENAME)

            # Prefer the noisy MS (thermalnoise was requested); fall back to
            # the noiseless MS if simobserve only produced that variant.
            ms_src_noisy = os.path.join(output_dir, _PROJECT_NAME,
                                        _PROJECT_NAME + '.noisy.ms')
            ms_src_clean = os.path.join(output_dir, _PROJECT_NAME,
                                        _PROJECT_NAME + '.ms')
            if os.path.isdir(ms_src_noisy):
                ms_src = ms_src_noisy
            elif os.path.isdir(ms_src_clean):
                ms_src = ms_src_clean
            else:
                raise RuntimeError(
                    'simobserve did not produce an MS under: '
                    + os.path.join(output_dir, _PROJECT_NAME))

            if os.path.isdir(ms_dst):
                shutil.rmtree(ms_dst)
            shutil.move(ms_src, ms_dst)

        finally:
            os.chdir(orig_dir)
            # Clean up the simobserve project directory and sky model
            sim_dir = os.path.join(output_dir, _PROJECT_NAME)
            if os.path.isdir(sim_dir):
                shutil.rmtree(sim_dir)
            sky_fits = os.path.join(output_dir, 'blank_sky.fits')
            if os.path.isfile(sky_fits):
                os.remove(sky_fits)

        return os.path.join(output_dir, _MS_FILENAME)

    def cleanup(self, output_dir):
        """
        Remove the MS previously created by create().

        Parameters
        ----------
        output_dir : str
            Same directory that was passed to create().
        """
        ms_path = os.path.join(os.path.abspath(output_dir), _MS_FILENAME)
        if os.path.isdir(ms_path):
            shutil.rmtree(ms_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_blank_sky(self, sky_fits_path, p):
        """
        Write a sky model FITS image for simobserve.

        simobserve's simutil computes scalefactor = inbright / nanmax(image).
        An all-zero image causes nanmax=0 → scalefactor=nan → "WARN: model is
        empty" and simobserve exits before writing any MS.

        The fix: place a single pixel with a negligibly small value (1e-10
        Jy/pixel) at the image centre.  This value is ~11 orders of magnitude
        below the typical thermal noise level (~0.05 Jy) so the resulting MS
        is effectively noise-only for all practical purposes.
        """
        try:
            from astropy.io import fits
            import numpy as np
        except ImportError:
            import pyfits as fits  # CASA 5.x environment
            import numpy as np

        n = p['image_size_pix']
        data = np.zeros((n, n), dtype=np.float32)

        # Non-zero seed pixel at the image centre to satisfy simobserve's
        # empty-model check.  Signal is negligible relative to thermal noise.
        cx, cy = n // 2, n // 2
        data[cy, cx] = 1e-10

        # Convert RA/Dec strings to degrees for the WCS header
        ra_deg  = self._ra_str_to_deg(p['ra'])
        dec_deg = self._dec_str_to_deg(p['dec'])

        hdr = fits.Header()
        hdr['SIMPLE']  = True
        hdr['BITPIX']  = -32
        hdr['NAXIS']   = 2
        hdr['NAXIS1']  = n
        hdr['NAXIS2']  = n
        hdr['CTYPE1']  = 'RA---SIN'
        hdr['CTYPE2']  = 'DEC--SIN'
        hdr['CRVAL1']  = ra_deg
        hdr['CRVAL2']  = dec_deg
        hdr['CRPIX1']  = n / 2.0
        hdr['CRPIX2']  = n / 2.0
        # Pixel scale in degrees (negative for RA axis, standard convention)
        hdr['CDELT1']  = -p['pixel_size_arcsec'] / 3600.0
        hdr['CDELT2']  =  p['pixel_size_arcsec'] / 3600.0
        hdr['CUNIT1']  = 'deg'
        hdr['CUNIT2']  = 'deg'
        hdr['BUNIT']   = 'Jy/pixel'
        hdr['EQUINOX'] = 2000.0

        hdu = fits.PrimaryHDU(data=data, header=hdr)
        hdu.writeto(sky_fits_path, overwrite=True)

    def _run_simobserve(self, sky_fits_path, p):
        """Call CASA's simobserve to produce the noisy MS."""
        # from phangsPipeline import casaStuff
        from casatasks import simobserve

        direction = 'J2000 {ra} {dec}'.format(ra=p['ra'], dec=p['dec'])
        total_bw_MHz = p['chan_width_MHz'] * p['nchan']

        # Do NOT pass inbright for a zero-flux model: inbright=0 causes
        # scalefactor = 0 / nanmax(zeros) = nan inside simutil, which triggers
        # "WARN: model is empty" and prevents the .noisy.ms from being written.
        simobserve(
            project        = _PROJECT_NAME,
            skymodel       = sky_fits_path,
            indirection    = direction,
            incell         = '{:.4f}arcsec'.format(p['pixel_size_arcsec']),
            incenter       = '{:.6f}GHz'.format(p['restfreq_GHz']),
            inwidth        = '{:.4f}MHz'.format(total_bw_MHz),
            setpointings   = True,
            integration    = '{:.1f}s'.format(p['integration_s']),
            totaltime      = '{:.1f}s'.format(p['total_time_s']),
            antennalist    = p['antennalist'],
            thermalnoise   = 'tsys-manual',
            overwrite      = True,
        )

    @staticmethod
    def _ra_str_to_deg(ra_str):
        """Convert 'HHhMMmSS.Ss' to decimal degrees."""
        ra_str = ra_str.strip()
        # Support 'HHhMMmSS.Ss' or 'HH:MM:SS.S'
        ra_str = ra_str.replace('h', ':').replace('m', ':').replace('s', '')
        parts = ra_str.split(':')
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m / 60.0 + s / 3600.0) * 15.0

    @staticmethod
    def _dec_str_to_deg(dec_str):
        """Convert '+/-DDdMMmSS.Ss' to decimal degrees."""
        dec_str = dec_str.strip()
        dec_str = dec_str.replace('d', ':').replace('m', ':').replace('s', '')
        sign = -1.0 if dec_str.startswith('-') else 1.0
        dec_str = dec_str.lstrip('+-')
        parts = dec_str.split(':')
        d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return sign * (d + m / 60.0 + s / 3600.0)
