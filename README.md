# photometry-sans-frustration (psf)

Interactive python wrapper for point-spread fitting (PSF) photometry using IRAF/DAOphot (dev version uses Astropy/photutils)

Requires two text files: one with transient coordinates and one with reference stars (see examples). If no reference star file is provided, code will attempt to create one by querying the PanSTARRS catalog.

psf:
measure transient photometry on image. Can specify multiple images with list or wildcards to run in batch mode (runs on all fits files in directory if left unspecified).

    psf.py
    psf.py -i image.fits [--options]

The other scripts below need to be re-written in Python 3 as some dependencies are no longer maintained. Use them at your own risk!

psfcom:
stack multiple images and then do photometry (NOTE: can now be done in python 3 version using psf.py --stack)

    psfcom.py --i images [--options]

psfsub:
spatially match, convolve and subtract template image, then do photometry. Wrapper for hotpants (A. Becker)

    psfsub.py -i image.fits -t template.fits [--options]

psfrerun.py:
measure photometry using coordinates and point-spread function from a previous run

    psfrerun.py -i image.fits -p psffile.fits [--options]