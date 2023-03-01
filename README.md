# photometry-sans-frustration (psf)

Interactive python wrapper for point-spread fitting (PSF) photometry using Astropy/photutils. Full python (including all dependencies)

To run: 
    python psf.py [--options]

Required inputs: 

  1. transient coordinates, either by -c [RA DEC] or a 1 line, 2-column tab-separated text file ending in '_coords.txt'. RA and Dec should be in decimal degrees

  2. list of coordinates of nearby stars (with magnitudes if zero point is required). This is in a file ending '_seq.txt'. If no sequence star file is provided, code will attempt to generate one automatically by querying catalogs (currently using PanSTARRS and SDSS). The format is: 
      
    ra dec g r i z

    336.7148649 17.1512778 19.188 18.454 18.153 18.031

Results will be saved in a directory called 'PSF_output' using a text file with a unique timestamp.

Full list of options available with -h. Some handy ones...


Specifying data / workflow:

  -i IMAGES : specify images to analyse (accepts wildcards). Otherwise runs on all fits images in directory
  
  -b BAND : specify bands to analyse (e.g. only use g band images)
  
  --quiet : no user prompts, run in the background
  
  --force : do not allow re-centroiding when performing final PSF fit to transient (generally not needed but useful if there is a bright nearby source or you have an exact position)
  
  
Pre-processing:

  --clean : cosmic ray cleaning with LACosmic
  
  --stack : stack images in same filter
  
  --time-bins N : cadence in days to use for stacked images (defaults to 1 day, i.e. stack from same night only)
  
  --overwrite-stacks : if not given and stacks already exist for desired band and time, will use those rather than making new ones
  
  
Image subtraction:
  
  --sub : perform template subtraction. Will subtract a local image called 'template_[band].fits' if it exists, otherwise will try to create one from PS1/SDSS
  
  --templatesize N: size of PS1/SDSS cutout to download in arcmin (default 5 arcmin)
  
  --cutoutsize : size of image to cutout for template subtraction in pixels
  

Calibration

  --queryrad N: area to search for sequence stars in PS1/SDSS (default 5 arcmin)
 
  --magmin / magmax MAG : faintest / brightest sequence stars to include in zero point calculation
  
  --aprad N : fixed aperture size in pixels to use for photometry (this is in addition to PSF photometry and optimal aperture photometry)
  
  --apfrac N.N : fraction of the PSF flux to include when calculating optimal aperture size (default is to use an aperture that captures 90% of PSF flux)
  
  --fwhm N : for instances where too few stars to build a PSF, give the FWHM in pixels to use for a simple Gaussian model
