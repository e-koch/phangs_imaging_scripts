"""
The PHANGS pipeline to handle post-processing of cubes. Works through
a single big class (the PostProcessHandler). This needs to be attached
to a keyHandler to access the target, product, and configuration
keys.

There should not be any direct calls to CASA in here. Eventually, this
should be able to run without CASA enabled (though it won't be able to
call any of the CASA-specific routines). Right now, just avoid direct
calls to CASA from this class.
"""

import os
import glob
import casaCubeRoutines as ccr
import casaMosaicRoutines as cmr
import casaFeatherRoutines as cfr

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

casa_enabled = True

class PostProcessHandler:
    """
    Class to handle post-processing of ALMA data. Post-processing here
    begins with the results of imaging and proceeds through reduced,
    science-ready data cubes.
    """

    def __init__(
        self,
        key_handler = None,
        dry_run = False,
        dochecks = True
        ):

        self._dochecks = dochecks
        
        self._targets_list = None
        self._mosaics_list = None
        self._line_products_list = None
        self._cont_products_list = None
        self._interf_configs_list = None
        self._feather_configs_list = None

        self._no_cont = False
        self._no_line = False

        self._feather_method = 'pbcorr'

        if key_handler is not None:
            self._kh = key_handler

        # Initialize the list variables
        self.set_targets(nobuild=True)
        self.set_mosaic_targets(nobuild=True)
        self.set_line_products(nobuild=True)
        self.set_cont_products(nobuild=True)
        self.set_interf_configs(nobuild=True)
        self.set_feather_configs(nobuild=True)

        self._build_lists()

        self.set_dry_run(dry_run)

        return(None)

#region Control what data gets processed

    def set_targets(
        self, 
        first=None, 
        last=None, 
        skip=[], 
        only=[],
        nobuild=False):
        """
        Set conditions on the list of targets to be considered. By
        default, consider all targets.
        """
        self._targets_first = first
        self._targets_last = last
        self._targets_skip = skip
        self._targets_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_mosaic_targets(
        self, 
        first=None, 
        last=None, 
        skip=[], 
        only=[],
        nobuild=False):
        """
        Set conditions on the list of mosaics to be considered. By
        default, consider all mosaics.
        """
        self._mosaics_first = first
        self._mosaics_last = last
        self._mosaics_skip = skip
        self._mosaics_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_line_products(
        self, 
        skip=[], 
        only=[], 
        nobuild=False,
        ):
        """
        Set conditions on the list of line products to be
        considered. By default, consider all products.
        """
        self._lines_skip = skip
        self._lines_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_cont_products(
        self, 
        skip=[], 
        only=[], 
        nobuild=False,
        ):
        """
        Set conditions on the list of continuum products to be
        considered. By default, consider all products.
        """
        self._cont_skip = skip
        self._cont_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_interf_configs(
        self, 
        skip=[], 
        only=[], 
        nobuild=False,
        ):
        """
        Set conditions on the list of interferometric array
        configurations to be considered. By default, consider all
        configurations.
        """
        self._interf_configs_skip = skip
        self._interf_configs_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_feather_configs(
        self, 
        skip=[], 
        only=[],
        nobuild=False,
        ):
        """
        Set conditions on the list of feathered array
        configurations to be considered. By default, consider all
        configurations.
        """
        self._feather_configs_skip = skip
        self._feather_configs_only = only

        if not nobuild:
            self._build_lists()
        return(None)

    def set_no_line(
        self,
        no_line = False):
        """
        Toggle the program to line products.
        """
        self._no_line = no_line
        self._build_lists()

    def set_no_cont(
        self,
        no_cont = False):
        """
        Toggle the program to skip continuum products.
        """
        self._no_cont = no_cont
        self._build_lists()

    def set_dry_run(
        self,
        dry_run = False):
        """
        Toggle the program using a 'dry run', i.e., not actually executing.
        """
        self._dry_run = dry_run

    def set_key_handler(
        self,
        key_handler = None):
        """
        Set the keyhandler being used by the pipeline.
        """
        self._kh = key_handler
        self._build_lists()

    def set_feather_method(
        self,
        method='pbcorr'
        ):
        """
        Set the approach to feathering used in the pipeline.
        """
        valid_choices = ['pbcorr','apodize']
        if method.lower() not in valid_choices:
            logger.error("Not a valid feather method: "+method)
            return(False)
        self._feather_method = method
        return(True)

#endregion

#region Behind the scenes infrastructure and book keeping.

    def _build_lists(
        self
        ):
        """
        Build the target lists.
        """

        if self._kh is None:
            logger.error("Cannot build lists without a keyHandler.")
            return(None)

        self._targets_list = self._kh.get_targets(            
            only = self._targets_only,
            skip = self._targets_skip,
            first = self._targets_first,
            last = self._targets_last,
            )

        self._mosaics_list = self._kh.get_linmos_targets(            
            only = self._mosaics_only,
            skip = self._mosaics_skip,
            first = self._mosaics_first,
            last = self._mosaics_last,
            )

        if self._no_line:
            self._line_products_list = []
        else:
            self._line_products_list = self._kh.get_line_products(
                only = self._lines_only,
                skip = self._lines_skip,
                )

        if self._no_cont:
            self._cont_products_list = []
        else:
            self._cont_products_list = self._kh.get_continuum_products(
                only = self._cont_only,
                skip = self._cont_skip,
                )

        self._interf_configs_list = self._kh.get_interf_configs(
            only = self._interf_configs_only,
            skip = self._interf_configs_skip,
            )

        self._feather_configs_list = self._kh.get_feather_configs(
            only = self._feather_configs_only,
            skip = self._feather_configs_skip,
            )

    def _all_products(
        self
        ):
        """
        Get a combined list of line and continuum products.
        """

        if self._cont_products_list is None:
            if self._line_products_list is None:
                return([])
            else:
                return(self._line_products_list)

        if self._line_products_list is None:
            if self._cont_products_list is None:
                return([])
            else:
                return(self._cont_products_list)

        if len(self._cont_products_list) is 0:
            if self._line_products_list is None:
                return ([])
            else:
                return(self._line_products_list)

        if len(self._line_products_list) is 0:
            if self._cont_products_list is None:
                return([])
            else:
                return(self._cont_products_list)
        
        return(self._line_products_list + self._cont_products_list)

#endregion

#region Master loops and file name routines

    def _fname_dict(
        self,
        target=None,
        config=None,
        product=None,
        extra_ext='',
        ):
        """
        Make the file name dictionary for all postprocess files given
        some target, config, product configuration.
        """

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Error checking
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        if target is None:
            logger.error("Need a target.")
            return()

        if product is None:
            logger.error("Need a product.")
            return()

        if config is None:
            logger.error("Need a config.")
            return()

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Initialize
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        fname_dict = {}

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Original files
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # Original cube and primary beam file
                    
        tag = 'orig'
        orig_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = None,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = orig_file
        
        tag = 'pb'
        pb_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = None,
            casa = True,
            casaext = '.pb')
        fname_dict[tag] = pb_file

        # Original single dish file (note that this comes with a
        # directory)

        has_sd = self._kh.has_singledish(target=target, product=product)
        tag = 'orig_sd'
        if has_sd:
            orig_sd_file = self._kh.get_sd_filename(
                target = target, product = product)            
            fname_dict[tag] = orig_sd_file
        else:
            fname_dict[tag] = None

        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
        # Processed files (apply the extra_ext tag here)
        # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

        # Primary beam corrected file

        tag = 'pbcorr'
        pbcorr_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'pbcorr'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = pbcorr_file

        # Files with round beams

        tag = 'round'
        round_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'round'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = round_file

        tag = 'pbcorr_round'
        pbcorr_round_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'pbcorr_round'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = pbcorr_round_file

        # Weight file for use in linear mosaicking

        tag = 'weight'
        weight_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'weight'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = weight_file

        tag = 'weight_aligned'
        weight_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'weight_aligned'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = weight_file

        # Common resolution parts for mosaic

        tag = 'linmos_commonres'
        commonres_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'linmos_commonres'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = commonres_file

        # Aligned parts for mosaic

        tag = 'linmos_aligned'
        aligned_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'linmos_aligned'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = aligned_file

        # Imported single dish file aligned to the interfometer data

        tag = 'prepped_sd'
        prepped_sd_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'singledish'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = prepped_sd_file

        # Singledish weight for use in linear mosaicking

        tag = 'sd_weight'
        sd_weight_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'singledish_weight'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = sd_weight_file 

        # Singledish data aliged to a common grid for mosaicking

        tag = 'sd_aligned'
        sd_align_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'singledish_aligned'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = sd_align_file 

        # Singledish weight for use in linear mosaicking now on a
        # common astrometric grid

        tag = 'sd_weight_aligned'
        sd_weight_aligned_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'singledish_weight_aligned'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = sd_weight_aligned_file 

        # Compressed files with edges trimmed off and smallest
        # reasonable pixel size.

        tag = 'trimmed'
        trimmed_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'trimmed'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = trimmed_file
        
        tag = 'pbcorr_trimmed'
        pbcorr_trimmed_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'pbcorr_trimmed'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = pbcorr_trimmed_file
        
        tag = 'trimmed_pb'
        trimmed_pb_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'trimmed'+extra_ext,
            casa = True,
            casaext = '.pb')
        fname_dict[tag] = trimmed_pb_file

        # Files converted to Kelvin, including FITS output files

        tag = 'trimmed_k'
        trimmed_k_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'trimmed_k'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = trimmed_k_file

        tag = 'trimmed_k_fits'
        trimmed_k_fits = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'trimmed_k'+extra_ext,
            casa = False)
        fname_dict[tag] = trimmed_k_fits
        
        tag = 'pbcorr_trimmed_k'
        pbcorr_trimmed_k_file = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'pbcorr_trimmed_k'+extra_ext,
            casa = True,
            casaext = '.image')
        fname_dict[tag] = pbcorr_trimmed_k_file

        tag = 'pbcorr_trimmed_k_fits'
        pbcorr_trimmed_k_fits = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'pbcorr_trimmed_k'+extra_ext,
            casa = False)
        fname_dict[tag] = pbcorr_trimmed_k_fits

        tag = 'trimmed_pb_fits'
        trimmed_pb_fits = self._kh.get_cube_filename(
            target = target, config = config, product = product,
            ext = 'trimmed_pb'+extra_ext,
            casa = False)
        fname_dict[tag] = trimmed_pb_fits

        # Return
        
        return(fname_dict)

#endregion

#region "Tasks" : Individual postprocessing steps

    def task_stage_interf_data(
        self,
        target = None,
        product = None,
        config = None,
        extra_ext_in = '',
        extra_ext_out = '',
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = self._kh.get_imaging_dir_for_target(target)
        outdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)
        fname_dict_out = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_out)
                
        # Copy the primary beam and the interferometric imaging
        
        for this_tag in ['orig', 'pb']:
            
            infile = fname_dict_in[this_tag]
            outfile = fname_dict_out[this_tag]
        
            # Check input file existence
            if check_files:
                if not (os.path.isdir(indir+infile)):
                    logger.warning("Missing "+indir+infile)
                    continue
    
            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Staging data for:")
            logger.info(str(target)+" , "+str(product)+" , "+str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")
            logger.info("Using ccr.copy_dropdeg.")
            logger.info("Staging "+outfile)
            
            if (not self._dry_run) and casa_enabled:
                ccr.copy_dropdeg(
                    infile=indir+infile,
                    outfile=outdir+outfile,
                    overwrite=True)

        return()

    def task_pbcorr(
        self,
        target = None,
        product = None,
        config = None,
        in_tag = 'orig',
        out_tag = 'pbcorr',
        extra_ext_in = '',
        extra_ext_out = '',
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = self._kh.get_postprocess_dir_for_target(target)
        outdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)
        fname_dict_out = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_out)

        infile = fname_dict_in[in_tag]
        outfile = fname_dict_out[out_tag]
        pbfile = fname_dict_in['pb']

        # Check input file existence
         
        if check_files:
            if not (os.path.isdir(indir+infile)):
                logger.warning("Missing "+indir+infile)
                return()
            if not (os.path.isdir(indir+pbfile)):
                logger.warning("Missing "+indir+pbfile)
                return()

        # Apply the primary beam correction to the data.
        
        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Primary beam correction for:")
        logger.info(str(target)+" , "+str(product)+" , "+str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")
        
        logger.info("Using ccr.primary_beam_correct")
        logger.info("Correcting to "+outfile)
        logger.info("Correcting from "+infile)
        logger.info("Correcting using "+pbfile)
        
        if (not self._dry_run) and casa_enabled:
            ccr.primary_beam_correct(
                infile=indir+infile,
                outfile=outdir+outfile,
                pbfile=indir+pbfile,
                overwrite=True)

        return()

    def task_round_beam(
        self,
        target = None,
        product = None,
        config = None,
        in_tag = 'pbcorr',
        out_tag = 'pbcorr_round',
        extra_ext_in = '',
        extra_ext_out = '',
        force_beam_as = None,
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = self._kh.get_postprocess_dir_for_target(target)
        outdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)
        fname_dict_out = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_out)
        
        infile = fname_dict_in[in_tag]
        outfile = fname_dict_out[out_tag]

        # Check input file existence        

        if check_files:
            if not (os.path.isdir(indir+infile)):
                logger.warning("Missing "+infile)
                return()

        # Convolve the data to have a round beam.
        
        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Convolving to a round beam for:")
        logger.info(str(target)+" , "+str(product)+" , "+str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")
        
        logger.info("Using ccr.convolve_to_round_beam")
        logger.info("Convolving from "+infile)
        logger.info("Convolving to "+outfile)
        if force_beam_as is not None:
            logger.info("Forcing beam to "+force_beam_as)
        
        if (not self._dry_run) and casa_enabled:
            ccr.convolve_to_round_beam(
                infile=indir+infile,
                outfile=outdir+outfile,
                force_beam=force_beam_as,
                overwrite=True)

        return()

    def task_stage_singledish(
        self,
        target = None,
        product = None,
        config = None,
        template_tag = 'pbcorr_round',
        out_tag = 'prepped_sd',
        extra_ext_in = '',
        extra_ext_out = '',
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = ''
        outdir = self._kh.get_postprocess_dir_for_target(target)
        tempdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)
        fname_dict_out = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_out)

        template = fname_dict_in[template_tag]
        infile = fname_dict_in['orig_sd']
        outfile = fname_dict_out[out_tag]

        # Check input file existence        
        
        if check_files:
            if (not (os.path.isdir(indir+infile))) and \
                    (not (os.path.isfile(indir+infile))):
                logger.warning("Missing "+infile)
                return()
            if not (os.path.isdir(tempdir+template)):
                logger.warning("Missing "+tempdir+template)
                return()

        # Stage the singledish data for feathering

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Preparing single dish data for:")
        logger.info(str(target)+" , "+str(product)+" , "+str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")
        
        logger.info("Using cfr.prep_sd_for_feather.")
        logger.info("Prepping "+outfile)
        logger.info("Original file "+infile)
        logger.info("Using interferometric template "+template)
        
        if (not self._dry_run) and casa_enabled:
            cfr.prep_sd_for_feather(
                sdfile_in=indir+infile,
                sdfile_out=outdir+outfile,
                interf_file=tempdir+template,
                do_import=True,
                do_dropdeg=True,
                do_align=True,
                do_checkunits=True,                                
                overwrite=True)

        return()

    def task_make_interf_weight(
        self,
        target = None,
        product = None,
        config = None,
        image_tag = 'pbcorr_round',
        in_tag = 'pb',
        input_type = 'pb',
        scale_by_noise = True,
        out_tag = 'weight',
        extra_ext_in = '',
        extra_ext_out = '',
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = self._kh.get_postprocess_dir_for_target(target)
        outdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)
        fname_dict_out = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_out)
        
        image_file = fname_dict_in[image_tag]
        infile = fname_dict_in[in_tag]
        outfile = fname_dict_out[out_tag]        

        # Check input file existence        
        
        if check_files:
            if not (os.path.isdir(indir+infile)):
                logger.warning("Missing "+infile)
                return()
            if not (os.path.isdir(indir+image_file)):
                logger.warning("Missing "+image_file)
                return()

        # Create a weight image for use linear mosaicking targets that
        # are part of a linear mosaic

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Making weight file for:")
        logger.info(str(target)+" , "+str(product)+" , "+str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")
        
        logger.info("Using cmr.generate_weight_file.")
        logger.info("Making weight file "+outfile)
        logger.info("Based off of primary beam file "+infile)
        logger.info("Measuring noise from file "+image_file)
                        
        if (not self._dry_run) and casa_enabled:
            cmr.generate_weight_file(
                image_file = indir+image_file,
                input_file = indir+infile,
                input_type = input_type,
                outfile = indir + outfile,
                scale_by_noise = scale_by_noise,
                overwrite=True)

        return()

    def task_make_singledish_weight(
        self,
        target = None,
        product = None,
        config = None,
        image_tag = 'prepped_sd',
        out_tag = 'sd_weight',
        extra_ext_in = '',
        extra_ext_out = '',
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = self._kh.get_postprocess_dir_for_target(target)
        outdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)
        fname_dict_out = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_out)
                        
        image_file = fname_dict_in[image_tag]
        outfile = fname_dict_out[out_tag]

        # Check input file existence        
    
        if check_files:
            if not (os.path.isdir(indir+image_file)):
                logger.warning("Missing "+image_file)
                return()

        # Make a weight file for single dish targets that
        # are part of a linear mosaic

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("Making single dish weight file for:")
        logger.info(str(target)+" , "+str(product)+" , "+str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&")
        logger.info("")
        
        logger.info("Using cmr.generate_weight_file.")
        logger.info("Making weight file "+outfile)
        logger.info("Measuring noise from file "+image_file)
            
        if (not self._dry_run) and casa_enabled:
            cmr.generate_weight_file(
                image_file = indir+image_file,
                input_value = 1.0,
                input_type = 'weight',
                outfile = indir + outfile,
                scale_by_noise = True,
                overwrite=True)
                
        return()

    def task_feather(
        self,
        target = None,
        product = None,
        config = None,
        interf_tag = 'pbcorr_round',
        sd_tag = 'prepped_sd',
        out_tag = 'pbcorr_round',
        extra_ext_in = '',
        extra_ext_out = '',
        apodize = False,
        apod_ext = 'pb',
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = self._kh.get_postprocess_dir_for_target(target)
        outdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)

        # Note that feather changes the config

        feather_config = self._kh.get_feather_config_for_interf_config(
            interf_config=config)

        fname_dict_out = self._fname_dict(
            target=target, config=feather_config, product=product, 
            extra_ext=extra_ext_out)
                        
        interf_file = fname_dict_in[interf_tag]
        sd_file = fname_dict_in[sd_tag]
        outfile = fname_dict_out[out_tag]

        # Error checking

        # Check input file existence        
    
        # Feather the single dish and interferometer data
                
        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Feathering interferometer and single dish data for:")
        logger.info(str(target)+" , "+str(product)+" , "+str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")
        
        logger.info("Using cfr.feather_two_cubes.")
        logger.info("Feathering "+outfile)
        logger.info("Feathering interferometric data "+interf_file)
        logger.info("Feathering single dish data "+sd_file)

        # Feather has a couple of algorithmic choices
        # associated with it. Run the method that the
        # user has selected.
        
        if apodize:

            apod_file = fname_dict_in[apod_ext]

            logger.info("Apodizing using file "+apod_file)

            if (not self._dry_run) and casa_enabled:
                cfr.feather_two_cubes(
                    interf_file=indir+interf_file,
                    sd_file=indir+sd_file,
                    out_file=outdir+outfile,
                    do_blank=True,
                    do_apodize=True,
                    apod_file=indir+apod_file,
                    apod_cutoff=0.0,
                    overwrite=True)
                
        else:
            
            if (not self._dry_run) and casa_enabled:                                
                cfr.feather_two_cubes(
                    interf_file=indir+interf_file,
                    sd_file=indir+sd_file,
                    out_file=outdir+outfile,
                    do_blank=True,
                    do_apodize=False,
                    apod_file=None,
                    apod_cutoff=-1.0,
                    overwrite=True)

        return()

    def task_compress(
        self,
        target = None,
        product = None,
        config = None,
        in_tag = 'pbcorr_round',
        out_tag = 'pbcorr_trimmed',
        do_pb_too = True,
        in_pb_tag = 'pb',
        out_pb_tag = 'pb_trimmed',
        extra_ext_in = '',
        extra_ext_out = '',
        apodize = False,
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = self._kh.get_postprocess_dir_for_target(target)
        outdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)
        fname_dict_out = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_out)

        infile = fname_dict_in['pbcorr_round']
        outfile = fname_dict_out['pbcorr_trimmed']

        infile_pb = fname_dict_in['pb']
        outfile_pb = fname_dict_out['trimmed_pb']

        # Check input file existence        

        if check_files:
            if not (os.path.isdir(indir+infile)):
                logger.warning("Missing "+infile)
                return()
            if not (os.path.isdir(indir+infile_pb)):
                logger.warning("Missing "+infile_pb)
                return()

        # Compress, reducing cube volume.

        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Trimming cube for:")
        logger.info(str(target)+" , "+str(product)+" , "+str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")

        logger.info("Producing "+outfile+" using ccr.trim_cube.")
        logger.info("Trimming from original file "+infile)
        
        if (not self._dry_run) and casa_enabled:
            ccr.trim_cube(
                infile=indir+infile,
                outfile=outdir+outfile,
                overwrite=True,
                inplace=False,
                min_pixperbeam=3)

        if do_pb_too is False:
            return()
            
        template = fname_dict_out['pbcorr_trimmed']

        if check_files:
            if not (os.path.isdir(outdir+template)):
                logger.warning("Missing "+template)
                return()

        logger.info("Aligning primary beam image to new astrometry")
        logger.info("Using ccr.align_to_target.")
        logger.info("Aligning original file "+infile_pb)
        logger.info("Aligning to produce output file "+outfile_pb)
        logger.info("Aligning to template "+template)

        if (not self._dry_run) and casa_enabled:
            ccr.align_to_target(
                infile=indir+infile_pb,
                outfile=outdir+outfile_pb,
                template=outdir+template,
                interpolation='cubic',
                overwrite=True,
                )

        return()

    def task_convert_units(
        self,
        target = None,
        product = None,
        config = None,
        in_tag = 'pbcorr_trimmed',
        out_tag = 'pbcorr_trimmed_k',
        extra_ext_in = '',
        extra_ext_out = '',
        apodize = False,
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = self._kh.get_postprocess_dir_for_target(target)
        outdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)
        fname_dict_out = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_out)
            
        infile = fname_dict_in[in_tag]
        outfile = fname_dict_out[out_tag]

        # Check input file existence        

        if check_files:
            if not (os.path.isdir(indir+infile)):
                logger.warning("Missing "+infile)
                return()

        # Change units from Jy/beam to Kelvin.
                        
        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Converting units for:")
        logger.info(str(target)+" , "+str(product)+" , "+str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")
        
        logger.info("Using ccr.convert_jytok")
        logger.info("Creating "+outfile)
        logger.info("Converting from original file "+infile)
        
        if (not self._dry_run) and casa_enabled:
            ccr.convert_jytok(
                infile=indir+infile,
                outfile=outdir+outfile,
                overwrite=True,
                inplace=False,
                )

        return()

    def task_export_to_fits(
        self,
        target = None,
        product = None,
        config = None,
        in_tag = 'pbcorr_trimmed',
        out_tag = 'pbcorr_trimmed_k',
        do_pb_too = True,
        in_pb_tag = 'trimmed_pb',
        out_pb_tag = 'trimmed_pb_fits',
        extra_ext_in = '',
        extra_ext_out = '',
        check_files = True,
        ):
        """
        """

        # Generate file names

        indir = self._kh.get_postprocess_dir_for_target(target)
        outdir = self._kh.get_postprocess_dir_for_target(target)
        fname_dict_in = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_in)
        fname_dict_out = self._fname_dict(
            target=target, config=config, product=product, extra_ext=extra_ext_out)
        
        infile = fname_dict_in[in_tag]
        outfile = fname_dict_out[out_tag]
        
        # Check input file existence        

        if check_files:
            if not (os.path.isdir(indir+infile)):
                logger.warning("Missing "+infile)
                return()

        # Export to FITS and clean up output
        
        logger.info("")
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("Exporting data to FITS and cleaning up cubes for:")
        logger.info(str(target)+" , "+str(product)+" , "+str(config))
        logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
        logger.info("")
        
        logger.info("Using ccr.export_and_cleanup.")
        logger.info("Export to "+outfile)
        logger.info("Writing from input cube "+infile)

        if (not self._dry_run) and casa_enabled:
            ccr.export_and_cleanup(
                infile=indir+infile,
                outfile=outdir+outfile,
                overwrite=True,
                remove_cards=[],
                add_cards=[],
                add_history=[],
                zap_history=True,
                round_beam=True,
                roundbeam_tol=0.01,
                )

        if do_pb_too is False:
            return()

        # Check input file existence        

        if check_files:
            if not (os.path.isdir(indir+infile_pb)):
                logger.warning("Missing "+infile_pb)
                return()

        infile_pb = fname_dict_in[in_pb_tag]
        outfile_pb = fname_dict_out[out_pb_tag]

        logger.info("Writing from primary beam "+infile_pb)
        logger.info("Writing output primary beam "+outfile_pb)
        
        if (not self._dry_run) and casa_enabled:
            ccr.export_and_cleanup(
                infile=indir+infile_pb,
                outfile=outdir+outfile_pb,
                overwrite=True,    
                remove_cards=[],
                add_cards=[],
                add_history=[],
                zap_history=True,
                round_beam=False,
                roundbeam_tol=0.01,
                )

        return()

    def _mosaic_one_target(
        self,
        target = None,
        product = None,
        config = None,
        do_convolve = True,
        do_align = True,
        do_singledish = True,
        do_mosaic = True,
        in_extra_ext = '',
        out_extra_ext = ''
        ):
        """
        Run the mosaicking steps: convolution, alignment, and linear
        mosaicking. Optionally also run them for the single dish data.
        """

        if target is None:
            logger.error("Need a target.")
            return()

        if product is None:
            logger.error("Need a product.")
            return()

        if config is None:
            logger.error("Need a config.")
            return()

        # Check that the target is a mosaic

        is_mosaic = self._kh.is_target_linmos(target)
        if is_mosaic == False:
            logger.error("Not a mosaic - "+str(target))
            return()

        # Get the relevant file names

        postprocess_dir = self._kh.get_postprocess_dir_for_target(target)
        
        fname_dict = self._fname_dict(
            target=target,
            product=product,
            config=config,
            extra_ext=out_extra_ext)

        mosaic_parts = self._kh.get_parts_for_linmos(target)

        # Check that all parts have single dish. Could change to allow
        # that only some parts need to have sd. Not 100% sure on what
        # we want.
        
        all_parts_have_sd = True
        for this_part in mosaic_parts:
            this_part_dict = self._fname_dict(
                target=this_part,
                config=config,
                product=product,
                extra_ext=in_extra_ext,
                )
            part_has_sd = self._kh.has_singledish(target=this_part, product=product)
            if part_has_sd is False:
                all_parts_have_sd = False

        if (not all_parts_have_sd) and do_singledish:
            logger.warning("Singledish processing requested but not all parts have single dish.")
            logger.warning("I am setting do_singledish to False.")
            do_singledish = False

        # Convolve the parts to a common resolution

        if do_convolve:
            
            indir = postprocess_dir
            outdir = postprocess_dir

            # Build the list of input files

            infile_list = []
            outfile_list = []

            for this_part in mosaic_parts:
                            
                this_part_dict = self._fname_dict(
                    target=this_part,
                    config=config,
                    product=product,
                    extra_ext=in_extra_ext,
                    )

                infile_list.append(indir+this_part_dict['pbcorr_round'])
                outfile_list.append(outdir+this_part_dict['linmos_commonres'])

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Convolving for mosaic for:")
            logger.info(str(target)+" , "+str(product)+" , "+str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using cmr.common_res_for_mosaic.")
            logger.info("Convolving "+target)
            logger.info("Convolving original files "+str(infile_list))
            logger.info("Convolving to convolved output "+str(outfile_list))
            
            # Allow overrides for the pixel padding (the
            # number of pixels added to the greatest
            # common beam for calculating the target
            # resolution) and the target resolution.
            
            pixel_padding = 2.0
            target_res = None
                        
            # TBD - check override dict for target
            # resolution and (I guess?) pixel padding.

            if not self._dry_run:
                cmr.common_res_for_mosaic(
                    infile_list = infile_list,
                    outfile_list = outfile_list,
                    do_convolve = True,
                    target_res = target_res,
                    pixel_padding = pixel_padding,
                    overwrite=True,
                    )

        # Generate a header and align data and weights to this new
        # astrometry for use in linear mosaicking.

        if do_align:

            indir = postprocess_dir
            outdir = postprocess_dir

            # Build the list of input files
            
            infile_list = []
            outfile_list = []

            # Get the input and output files for individual
            # parts. Also include the weights here.

            for this_part in mosaic_parts:
                            
                this_part_dict = self._fname_dict(
                    target=this_part,
                    config=config,
                    product=product,
                    extra_ext=in_extra_ext,
                    )
                
                infile_list.append(indir+this_part_dict['linmos_commonres'])
                outfile_list.append(outdir+this_part_dict['linmos_aligned'])

                infile_list.append(indir+this_part_dict['weight'])
                outfile_list.append(outdir+this_part_dict['weight_aligned'])
                
            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Aligning for mosaic for:")
            logger.info(str(target)+" , "+str(product)+" , "+str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")
            
            logger.info("Using cmr.common_grid_for_mosaic.")
            logger.info("Aligning "+target)
            logger.info("Convolving original files "+str(infile_list))
            logger.info("Convolving to convolved output "+str(outfile_list))

            # TBD implement overrides
            
            ra_ctr = None 
            dec_ctr = None
            delta_ra = None 
            delta_dec = None
                        
            if not self._dry_run:
                cmr.common_grid_for_mosaic(
                    infile_list = infile_list,
                    outfile_list = outfile_list,
                    ra_ctr = ra_ctr, 
                    dec_ctr = dec_ctr,
                    delta_ra = delta_ra, 
                    delta_dec = delta_dec,
                    allow_big_image = False,
                    too_big_pix=1e4,   
                    asvelocity=True,
                    interpolation='cubic',
                    axes=[-1],
                    overwrite=True,
                    )

        if do_align and do_singledish:

            indir = postprocess_dir
            outdir = postprocess_dir

            # Build the list of input files

            infile_list = []
            outfile_list = []

            # Get the input and output files for individual
            # parts. Also include the weights here.

            for this_part in mosaic_parts:
                            
                this_part_dict = self._fname_dict(
                    target=this_part,
                    config=config,
                    product=product,
                    extra_ext=in_extra_ext,
                    )

                infile_list.append(indir+this_part_dict['prepped_sd'])
                outfile_list.append(outdir+this_part_dict['sd_aligned'])

                infile_list.append(indir+this_part_dict['sd_weight'])
                outfile_list.append(outdir+this_part_dict['sd_weight_aligned'])

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Aligning single dish for mosaic for:")
            logger.info(str(target)+" , "+str(product)+" , "+str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using cmr.common_grid_for_mosaic.")
            logger.info("Aligning "+target)
            logger.info("Convolving original files "+str(infile_list))
            logger.info("Convolving to convolved output "+str(outfile_list))

            # TBD - implement use of interferometer grid as a template.
            
            # TBD implement overrides

            ra_ctr = None 
            dec_ctr = None
            delta_ra = None 
            delta_dec = None
            
            if not self._dry_run:
                cmr.common_grid_for_mosaic(
                    infile_list = infile_list,
                    outfile_list = outfile_list,
                    ra_ctr = ra_ctr, 
                    dec_ctr = dec_ctr,
                    delta_ra = delta_ra, 
                    delta_dec = delta_dec,
                    allow_big_image = False,
                    too_big_pix=1e4,   
                    asvelocity=True,
                    interpolation='cubic',
                    axes=[-1],
                    overwrite=True,
                    )

        # Execute linear mosaicking for the interferometer data

        if do_mosaic:

            indir = postprocess_dir
            outdir = postprocess_dir
            outfile = fname_dict['pbcorr_round']

            infile_list = []
            weightfile_list = []

            # Get the input and weight files for
            # individual parts.

            for this_part in mosaic_parts:
                
                this_part_dict = self._fname_dict(
                    target=this_part,
                    config=config,
                    product=product,
                    extra_ext=in_extra_ext,
                    )
                
                infile_list.append(indir+this_part_dict['linmos_aligned'])
                weightfile_list.append(indir+this_part_dict['weight_aligned'])

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Executing linear mosaic for:")
            logger.info(str(target)+" , "+str(product)+" , "+str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")

            logger.info("Using cmr.mosaic_aligned_data.")
            logger.info("Creating "+outfile)
            logger.info("Mosaicking original files "+str(infile_list))
            logger.info("Weighting by "+str(weightfile_list))
            
            if not self._dry_run:
                cmr.mosaic_aligned_data(
                    infile_list = infile_list,
                    weightfile_list = weightfile_list,
                    outfile = outdir+outfile,
                    overwrite=True)
                
        # Execute linear mosaicking for the single dish data

        if do_mosaic and do_singledish:

            indir = postprocess_dir
            outdir = postprocess_dir
            outfile = fname_dict['prepped_sd']

            infile_list = []
            weightfile_list = []

            # Get the input and weight files for
            # individual parts.

            for this_part in mosaic_parts:
                            
                this_part_dict = self._fname_dict(
                    target=this_part,
                    config=config,
                    product=product,
                    extra_ext=in_extra_ext,
                    )

                infile_list.append(indir+this_part_dict['sd_aligned'])
                weightfile_list.append(indir+this_part_dict['sd_weight_aligned'])

            logger.info("")
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("Executing linear mosaic for single dish for:")
            logger.info(str(target)+" , "+str(product)+" , "+str(config))
            logger.info("&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%")
            logger.info("")
            
            logger.info("Using cmr.mosaic_aligned_data.")
            logger.info("Creating "+outfile)
            logger.info("Mosaicking original files "+str(infile_list))
            logger.info("Weighting by "+str(weightfile_list))

            if not self._dry_run:
                cmr.mosaic_aligned_data(
                    infile_list = infile_list,
                    weightfile_list = weightfile_list,
                    outfile = outdir+outfile,
                    overwrite=True)
                
        return()
            
#endregion

#region Recipes
    
    def recipe_prep_one_target(
        self,
        target = None,
        product = None,
        config = None,
        check_files = True,
        ):
        """
        """

        # Work out file names and note whether the target is part of a
        # mosaic, has single dish data, etc.

        fname_dict = self._fname_dict(
            target=target, product=product, config=config)

        imaging_dir = self._kh.get_imaging_dir_for_target(target)
        has_imaging = os.path.isdir(imaging_dir + fname_dict['orig'])
        has_singledish = self._kh.has_singledish(target=target, product=product)        
        is_part_of_mosaic = self._kh.is_target_in_mosaic(target)

        if not has_imaging:
            logger.warning("No imaging for "+fname_dict['orig']+". Returning.")
            return()

        # Call tasks

        self.task_stage_interf_data(
            target=target, config=config, product=product,
            check_files=check_files
            )

        self.task_pbcorr(
            target=target, config=config, product=product,
            check_files=check_files
            )

        self.task_round_beam(
            target=target, config=config, product=product,
            check_files=check_files
            )

        if has_singledish:
            self.task_stage_singledish(
                target=target, config=config, product=product,
                check_files=check_files
                )

        if is_part_of_mosaic:
            self.task_make_interf_weight(
                target=target, config=config, product=product,
                check_files=check_files, scale_by_noise=True,
                )

        if is_part_of_mosaic and has_singledish:
            self.task_make_singledish_weight(
                target=target, config=config, product=product,
                check_files=check_files,
                )

        return()

    def recipe_cleanup_one_target(
        self,
        target = None,
        product = None,
        config = None,
        check_files = True,
        ext_ext = '',
        ):
        """
        """

        self.task_compress(
            target=target, config=config, product=product,
            check_files=check_files, apodize=apodize, do_pb_too=True,
            extra_ext_in=ext_ext_in, extra_ext_out=ext_ext_out,
            )

        self.task_export_to_fits(
            target=target, config=config, product=product,
            check_files=check_files, do_pb_too=True,
            extra_ext_in=ext_ext_in, extra_ext_out=ext_ext_out,
            )

        return()        

    

#endregion

#region Loops

    def loop_postprocess(
        self,
        do_prep=False,
        do_feather=False,
        do_mosaic=False,
        do_cleanup=False,
        feather_apod=False,
        feather_noapod=False,
        ):
        """
        """

        if self._targets_list is None:            
            logger.error("Need a target list.")
            return(None)
 
        if self._all_products is None:            
            logger.error("Need a products list.")
            return(None)

        # Prepare the interferometer data that has imaging. Includes
        # staging the single dish data, making weights, etc.
        
        if do_prep:
    
            for this_target in self._targets_list:

                for this_product in self._all_products():
                    
                    for this_config in self._interf_configs_list:
                       
                        fname_dict = self._fname_dict(
                            target=this_target, product=this_product, config=this_config)
                        
                        imaging_dir = self._kh.get_imaging_dir_for_target(this_target)
                        has_imaging = os.path.isdir(imaging_dir + fname_dict['orig'])
                        has_singledish = self._kh.has_singledish(target=this_target, product=this_product)        
                        is_part_of_mosaic = self._kh.is_target_in_mosaic(this_target)

                        if not has_imaging:
                            logger.debug("Skipping "+this_target+" because it lacks imaging.")
                            logger.debug(imaging_dir+fname_dict['orig'])
                            continue

                        self.recipe_prep_one_target(
                            target = this_target, product = this_product, config = this_config,
                            check_files = True)

        # Feather the interferometer configuration data that has
        # imaging. We'll return to mosaicks in the next steps.
                        
        if do_feather:
            
            for this_target in self._targets_list:

                for this_product in self._all_products():
                    
                    for this_config in self._interf_configs_list:

                        fname_dict = self._fname_dict(
                            target=this_target, product=this_product, config=this_config)
                        
                        imaging_dir = self._kh.get_imaging_dir_for_target(this_target)
                        has_imaging = os.path.isdir(imaging_dir + fname_dict['orig'])
                        has_singledish = self._kh.has_singledish(target=this_target, product=this_product)        
                        is_part_of_mosaic = self._kh.is_target_in_mosaic(this_target)
                            
                        if not has_imaging:
                            logger.debug("Skipping "+this_target+" because it lacks imaging.")
                            logger.debug(imaging_dir+fname_dict['orig'])
                            continue
                            
                        if not has_singledish:
                            logger.debug("Skipping "+this_target+" because it lacks single dish.")
                            continue

                        if feather_apod:                            
                            self.task_feather(
                                target = this_target, product = this_product, config = this_config,
                                apodize=True, apod_ext='pb',extra_ext_out='_apod',check_files=True,
                                )

                        if feather_noapod:
                            self.task_feather(
                                target = this_target, product = this_product, config = this_config,
                                apodize=False, extra_ext_out='',check_files=True,
                                )

        if do_mosaic:

            for this_target in self._targets_list:
                
                is_mosaic = self._kh.is_target_linmos(this_target)
                if not is_mosaic:
                    continue

                for this_product in self._all_products():
                    
                    for this_config in self._interf_configs_list:

                        pass

        if do_feather:
            
            for this_target in self._targets_list:

                for this_product in self._all_products():
                    
                    for this_config in self._interf_configs_list:

                        pass
                        
        if do_cleanup:
            
            for this_target in self._targets_list:

                for this_product in self._all_products():
                    
                    all_configs = []
                    for this_config in self._interf_configs_list:
                        all_configs.append(this_config)
                    for this_config in self._feather_configs_list:
                        all_configs.append(this_config)

                    for this_config in all_configs:
                        
                        self.recipe_cleanup_one_target(
                            target = this_target, product = this_product, config = this_config,
                            check_files = True)

#endregion
