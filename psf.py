#!/usr/bin/env python

version = '1.4'

'''
    PSF: PHOTOMETRY SANS FRUSTRATION

    Written by Matt Nicholl, 2015-2022

    Requirements:

    Needs photutils, astropy, numpy, matplotlib, skimage, requests, astroquery, astroalign.
    Also pyzogy if running with template subtraction.

    Previously in IRAF, completely re-written for Python 3 using photutils

    Run in directory with image files in FITS format.

    Use -i flag to specify image names separated by spaces. If no names given,
    assumes you want to do photometry on every image in directory.

    File with SN coordinates must exist in same directory or parent
    directory, and should be named *_coords.txt
    Format:
    RA_value  DEC_value

    File with local sequence star coordinates (J2000) and magnitudes
    may exist in same directory or parent directory, and should be
    named *_seq.txt
    Format:
    RA_value DEC_value  MAGBAND1_value    MAGBAND2_value    ...

    If sequence star file does not exist, code will create one from PS1 or SDSS archive.
    But note that PS1/SDSS only contain ugrizy magnitudes!

    Given this list of field star magnitudes and coordinates, psf.py will
    compute the zeropoint of the image, construct a point spread function
    from these stars, fit this to the target of interest, show the resulting
    subtraction, and return the apparent magnitude from both PSF and aperture
    photometry

    Run with python psf.py <flags> (see help message with psf.py --help)

    Outputs a text file PSF_phot_X.txt (where X is a unique timestamp, to avoid overwriting previous results)
    Format of text file is:
        image  filter  mjd  PSFmag  err  APmag  err  comments
        - Row exists for each input image used in run
        - PSFmag is from PSF fitting, APmag is from simple aperture photometry
        using aperture size specified with --ap (default 10 pixel radius)
        - ZP is measured from sequence stars
        - Any template subtracted
        - comment allows user to specify if e.g. PSF fit looked unreliable
        
        
    For image subtraction, template name must be template_<standard filter name>.fits !!!

    NEED TO FULLY DOCUMENT...

    '''

import numpy as np
import glob
import astropy
import photutils
import sys
import shutil
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
import argparse
from matplotlib.patches import Circle
import requests
from astropy import visualization
from skimage.registration import phase_cross_correlation, optical_flow_tvl1, optical_flow_ilk
from skimage.transform import warp
from scipy.ndimage import interpolation as interp
import time
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import astropy.units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
import astroalign as aa
from reproject import reproject_interp
from PyZOGY.subtract import run_subtraction
from photutils.utils import calc_total_error
from photutils.psf import IntegratedGaussianPRF
from ccdproc import cosmicray_lacosmic as lacosmic
import warnings


warnings.filterwarnings("ignore")



# Optional flags:

parser = argparse.ArgumentParser()

parser.add_argument('--ims','-i', dest='file_to_reduce', default='', nargs='+',
                    help='List of files to reduce (accepts wildcards or '
                    'space-delimited list)')

parser.add_argument('--bands','-b', dest='bands', default='', nargs='+',
                    help='List of bands to use (space-delimited list), will skip others if present')

parser.add_argument('--coords','-c', dest='coords', default=[None,None], nargs=2, type=float,
                    help='Coordinates of target')

parser.add_argument('--magmin', dest='magmin', default=20, type=float,
                    help='Faintest sequence stars to use (stars below 3 sigma will be removed anyway)')

parser.add_argument('--magmax', dest='magmax', default=16, type=float,
                    help='Brightest sequence stars to use')

parser.add_argument('--shifts', dest='shifts', default=False, action='store_true',
                    help='Apply manual shifts if WCS is a bit off')

parser.add_argument('--aprad', dest='aprad', default=10, type=int,
                    help='Radius for aperture photometry')

parser.add_argument('--apfrac', dest='apfrac', default=0.9, type=float,
                    help='Fraction of PSF flux to include in optimal aperture')

parser.add_argument('--stamprad', dest='stamprad', default=15, type=int,
                    help='Radius for PSF extraction')
                    
parser.add_argument('--skyrad', dest='skyrad', default=5, type=int,
                    help='Width of annulus for sky background')

parser.add_argument('--box', dest='bkgbox', default=200, type=int,
                    help='Size of stamps for background fit')

parser.add_argument('--psfthresh', dest='psfthresh', default=20., type=float,
                    help='SNR threshold for inclusion in PSF model')

parser.add_argument('--sigma', dest='sigma_gauss', default=2., type=float,
                    help='Sigma for basic Gaussian PSF (used when insufficient stars)')

parser.add_argument('--zpsig', dest='sigClip', default=1, type=int,
                    help='Sigma clipping for rejecting sequence stars')

parser.add_argument('--samp', dest='samp', default=1, type=int,
                    help='Oversampling factor for PSF build')

parser.add_argument('--quiet', dest='quiet', default=False, action='store_true',
                    help='Run with no user prompts')

parser.add_argument('--stack', dest='stack', default=False, action='store_true',
                    help='Stack images that are in the same filter')

parser.add_argument('--timebins', dest='bin', default=1.0, type=float,
                    help='Width of bins for stacking (in days)')

parser.add_argument('--clean', dest='clean', default=False, action='store_true',
                    help='Clean images with lacosmic')

parser.add_argument('--sub', dest='sub', default=False, action='store_true',
                    help='Subtract template images')

parser.add_argument('--noalign', dest='noalign', default=False, action='store_true',
                    help='Do not align template to image (use if already aligned)')

parser.add_argument('--cut', dest='cut', default=600, type=int,
                    help='Cutout size for image subtraction')

parser.add_argument('--sci-sat', dest='sci_sat', default=50000, type=int,
                    help='Max valid science pixel value for image subtraction')

parser.add_argument('--tmpl-sat', dest='tmpl_sat', default=500000, type=int,
                    help='Max valid template pixel value for image subtraction')

parser.add_argument('--keep', dest='keep', default=False, action='store_true',
                    help='Keep intermediate products')

parser.add_argument('--force', dest='forcepos', default=False, action='store_true',
                    help='Do not recenter on transient')

args = parser.parse_args()

magmin = args.magmin
magmax = args.magmax

if magmin < magmax:
    print('error: magmin brighter than magmax - resetting to defaults')
    magmin = 22
    magmax = 16.5

shifts = args.shifts
aprad = args.aprad
apfrac = args.apfrac
stamprad0 = args.stamprad
skyrad = args.skyrad
bkgbox = args.bkgbox
psfthresh0 = args.psfthresh
sigma_gauss_0 = args.sigma_gauss
sigClip = args.sigClip
samp0 = args.samp
quiet = args.quiet
stack = args.stack
timebins = args.bin
clean = args.clean
sub = args.sub
cutoutsize = args.cut
noalign = args.noalign
tmpl_sat = args.tmpl_sat
sci_sat = args.sci_sat
keep = args.keep
forcepos = args.forcepos

ims = [i for i in args.file_to_reduce]

bands = [i for i in args.bands]

local_template_list = glob.glob('template_*.fits')

coords1 = [i for i in args.coords]

# If no images provided, run on all images in directory
if len(ims) == 0:
    ims = glob.glob('*.fits')
    if 'tmpl_aligned.fits' in ims:
        ims.remove('tmpl_aligned.fits')
    if 'tmpl_trim.fits' in ims:
        ims.remove('tmpl_trim.fits')
    if 'tmpl_psf.fits' in ims:
        ims.remove('tmpl_psf.fits')
    if 'sci_trim.fits' in ims:
        ims.remove('sci_trim.fits')
    if 'sci_psf.fits' in ims:
        ims.remove('sci_psf.fits')
    if 'sub.fits' in ims:
        ims.remove('sub.fits')
    for i in local_template_list:
        if i in ims:
            ims.remove(i)



##################################################



##### FUNCTIONS TO QUERY PANSTARRS #######

def PS1catalog(ra,dec,magmin=25,magmax=8):

    queryurl = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/stack?'
    queryurl += 'ra='+str(ra)
    queryurl += '&dec='+str(dec)
    queryurl += '&radius=0.083'
    queryurl += '&columns=[raStack,decStack,gPSFMag,rPSFMag,iPSFMag,zPSFMag,yPSFMag,iKronMag]'
    queryurl += '&nDetections.gte=6&pagesize=10000'

    print('\nQuerying PS1 for reference stars via MAST...\n')

    query = requests.get(queryurl)

    results = query.json()

    if len(results['data']) > 1:
    
        data = np.array(results['data'])

        # Star-galaxy separation: star if PSFmag - KronMag < 0.1
        data = data[:,:-1][data[:,4]-data[:,-1]<0.1]
        

        # Below is a bit of a hack to remove duplicates
        catalog = coords.SkyCoord(ra=data[:,0]*u.degree, dec=data[:,1]*u.degree)
        
        data2 = []
        
        indices = np.arange(len(data))
        
        used = []
        
        for i in data:
            source = coords.SkyCoord(ra=i[0]*u.degree, dec=i[1]*u.deg)
            d2d = source.separation(catalog)
            catalogmsk = d2d < 2.5*u.arcsec
            indexmatch = indices[catalogmsk]
            for j in indexmatch:
                if j not in used:
                    data2.append(data[j])
                    for k in indexmatch:
                        used.append(k)


        np.savetxt('PS1_seq.txt',data2,fmt='%.8f\t%.8f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f', header='ra\tdec\tg\tr\ti\tz\ty',comments='')

        print('Success! Sequence star file created: PS1_seq.txt')

    else:
        sys.exit('Field not in PS1! Exiting')



def PS1cutouts(ra,dec,filt):

    print('\nSearching for PS1 images of field...\n')

    ps1_url = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?'

    ps1_url += '&ra='+str(ra)
    ps1_url += '&dec='+str(dec)
    ps1_url += '&filters='+filt

    ps1_im = requests.get(ps1_url)

    try:
        image_name = ps1_im.text.split()[17]

        print('Image found: ' + image_name + '\n')

        cutout_url = 'http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?&filetypes=stack&size=2500'

        cutout_url += '&ra='+str(ra)
        cutout_url += '&dec='+str(dec)
        cutout_url += '&filters='+filt
        cutout_url += '&format=fits'
        cutout_url += '&red='+image_name

        dest_file = 'template_'+filt+'.fits'

        cmd = 'wget -O %s "%s"' % (dest_file, cutout_url)

        os.system(cmd)

        print('Template downloaded as ' + dest_file + '\n')

    except:
        print('\nPS1 template search failed!\n')

        dest_file = ''
        
    return dest_file


##################################


# QUERY SDSS

def SDSScutouts(ra,dec,filt):

    print('\nSearching for SDSS images of field...\n')

    pos = coords.SkyCoord(str(ra)+' '+str(dec), unit='deg', frame='icrs')
    
    try:
        xid = SDSS.query_region(pos)
        if len(xid)>1:
            xid.remove_rows(slice(1,len(xid)))
            
        im = SDSS.get_images(matches=xid, band=filt)
        
        dest_file = 'template_'+filt+'.fits'
        
        im[0].writeto(dest_file)
        
        print('Template downloaded as ' + dest_file + '\n')

    except:
        print('SDSS template search failed!\n')
        
        dest_file = ''
        
    return dest_file
        
        
def SDSScatalog(ra,dec,magmin=25,magmax=8):
 
    print('\nQuerying SDSS for reference stars via Astroquery...\n')

    pos = coords.SkyCoord(str(ra)+' '+str(dec), unit='deg', frame='icrs')
    
    data = SDSS.query_region(pos,radius='0.1d',fields=('ra','dec','u','g','r','i','z','type'))
    
    data = data[data['type'] == 6]

    
    # NEED TO REMOVE DUPLICATES

    catalog = coords.SkyCoord(ra=data['ra']*u.degree, dec=data['dec']*u.degree)
    
    data2 = data[:0].copy()
    
    indices = np.arange(len(data))
    
    used = []
    
    for i in data:
        source = coords.SkyCoord(ra=i['ra']*u.degree, dec=i['dec']*u.deg)
        d2d = source.separation(catalog)
        catalogmsk = d2d < 2.5*u.arcsec
        indexmatch = indices[catalogmsk]
        for j in indexmatch:
            if j not in used:
                data2.add_row(data[j])
                for k in indexmatch:
                    used.append(k)

    data2.write('SDSS_seq.txt',format='ascii',overwrite=True, exclude_names=['type'],delimiter='\t')
    
    print('Success! Sequence star file created: SDSS_seq.txt')


# Try to match header keyword to a known filter automatically:

filtSyn = {'u':['u','SDSS-U','up','up1','U640','F336W','Sloan_u','u_Sloan'],
           'g':['g','SDSS-G','gp','gp1','g782','F475W','g.00000','Sloan_g','g_Sloan'],
           'r':['r','SDSS-R','rp','rp1','r784','F625W','r.00000','Sloan_r','r_Sloan'],
           'i':['i','SDSS-I','ip','ip1','i705','F775W','i.00000','Sloan_i','i_Sloan'],
           'z':['z','SDSS-Z','zp','zp1','z623','zs', 'F850LP','z.00000','Sloan_z','z_Sloan'],
           'J':['J'],
           'H':['H'],
           'K':['K','Ks'],
           'U':['U','U_32363A'],
           'B':['B','B_39692','BH'],
           'V':['V','V_36330','VH'],
           'R':['R','R_30508'],
           'I':['I','I_36283']}

# UPDATE WITH QUIRKS OF MORE TELESCOPES...

filtAll = 'u,g,r,i,z,U,B,V,R,I,J,H,K'



print('#################################################\n#                                               #\n#  Welcome to PSF: Photometry Sans Frustration  #\n#                    (V'+version+')                     #\n#      Written by Matt Nicholl (2015-2022)      #\n#                                               #\n#################################################')


# A place to save output files
outdir = 'PSF_output'

if not os.path.exists(outdir): os.makedirs(outdir)

# A file to write final magnitudes
results_filename = os.path.join(outdir,'PSF_phot_'+str(int(time.time()))+'.txt')
outFile = open(results_filename,'w')
outFile.write('#image\ttarget\tfilter\tmjd\tPSFmag\terr\tAp_opt\terr\tAp_big\terr\tap_limit\tdet_limit\tZP\terr\ttemplate\tcomments')



# Prepare to plot images and PSFs

plt.figure(1,(14,7))
plt.ion()
plt.show()



################################
# Part one: get coordinates
################################

# Search for SN coordinates file (1 line: RA Dec) in this directory and parents

if coords1[0] and coords1[1]:
    RAdec = np.array(coords1)
    
else:
    suggSn = glob.glob('*coords.txt')
    if len(suggSn)==0:
        suggSn = glob.glob('../*coords.txt')
    if len(suggSn)==0:
        suggSn = glob.glob('../../*coords.txt')

    # Needs SN coords to run!
    if len(suggSn)>0:
        snFile = suggSn[0]
    else:
        sys.exit('No SN coordinates. Please supply via -c RA DEC or *_coords.txt')

    print('\n####################\n\nSN coordinates found: '+snFile)

    RAdec = np.genfromtxt(snFile)




#### BEGIN LOOP OVER IMAGES ####

# In case wcs offset is needed
x_sh_1 = 0
y_sh_1 = 0



################################
# Part two: image header info
################################
usedfilters = []
filtertab = []

for image in ims:

    if not os.path.exists(image):
        continue

    print('\nFile: ' + image)
    try:
        filtername = fits.getval(image,'FILTER')
    except:
        try:
            filtername = fits.getval(image,'FILTER1')
            if filtername == ('air' or 'none' or 'clear'):
                filtername = fits.getval(image,'FILTER2')
            if filtername == ('air' or 'none' or 'clear'):
                filtername = fits.getval(image,'FILTER3')
        except:
            try:
                filtername = fits.getval(image,'NCFLTNM2')
            except:
                filtername = 'none'
    print('Filter found in header: ' + filtername)
    if filtername=='none':
        filtername = input('Please enter filter ('+filtAll+') ')

    for j in filtSyn:
        if filtername in filtSyn[j]:
            filtername = j
            print('Calibrating to filter: ' + filtername)

    if filtername not in filtAll:
        filtername = input('Please enter filter ('+filtAll+') ')
        
    if filtername in bands or len(bands) == 0:

        try:
            mjd = fits.getval(image,'MJD')
        except:
            try:
                mjd = fits.getval(image,'MJD-OBS')
            except:
                try:
                    mjd = fits.getval(image,'OBSMJD')
                except:
                    try:
                        jd = fits.getval(image,'JD')
                        mjd = jd - 2400000.5
                    except:
                        mjd = input('No MJD found, please enter: [99999] ')
                        if not mjd: mjd = 99999

        mjd = float(mjd)

        template = ''

        if sub==True:
            if os.path.exists('template_'+filtername+'.fits'):
                template = 'template_'+filtername+'.fits'
            elif os.path.exists('../template_'+filtername+'.fits'):
                template = '../template_'+filtername+'.fits'
            elif os.path.exists('../../template_'+filtername+'.fits'):
                template = '../../template_'+filtername+'.fits'
            else:
                print('No template found locally...')
                if filtername in ('g','r','i','z'):
                    try:
                        template = PS1cutouts(RAdec[0],RAdec[1],filtername)
                    except:
                        print('Error: could not match template from PS1')
                        template = ''
                elif filtername == 'u':
                    try:
                        template = SDSScutouts(RAdec[0],RAdec[1],filtername)
                    except:
                        print('Error: could not match template from SDSS')
                        template = ''


        filtertab.append([image, filtername, mjd, template])

        if not filtername in usedfilters:
            usedfilters.append(filtername)
            
    else:
        print('Band outside user-specified list, skipping!')
        continue

filtertab = np.array(filtertab)



#################################
# Part three: match templates and
#          stack images if needed
#################################

for f in usedfilters:

    print('\n\n#########\n'+f+'-band\n#########')
    
    filtertab2 = []

    ims1 = filtertab[filtertab[:,1]==f]
    
    templates1 = filtertab[:,-1][filtertab[:,1]==f]
        
    has_template = False
        
    if sub==True:
        if len(np.unique(templates1)) > 1:
            print('Warning: different templates for same filter!')
            print('Using '+templates1[0])
        
        template = templates1[0]
        
        if len(template) > 0:
            has_template = True

    ######## Search for sequence star file (RA, dec, mags)

    # Check folder/parent for sequence stars
    suggSeq = glob.glob('*seq.txt')
    if len(suggSeq)==0:
        suggSeq = glob.glob('../*seq.txt')
    if len(suggSeq)==0:
        suggSeq = glob.glob('../../*seq.txt')
        
    # Note PS1 sequence to avoid always downloading
    if 'PS1_seq.txt' in suggSeq:
        hasPS1 = True
    else:
        hasPS1 = False

    # If more than one sequence star file, check which has the required filter
    if len(suggSeq)>0:
        n = 0
        for i in range(len(suggSeq)):
            seqDat = np.genfromtxt(suggSeq[i-n])
            seqHead = np.genfromtxt(suggSeq[i-n],skip_footer=len(seqDat)-1,dtype=str)
            if f not in seqHead:
                suggSeq.pop(i-n)
                n += 1
                
    # if PS1 has required filter, give it priority
    if 'PS1_seq.txt' in suggSeq:
        seqFile = 'PS1_seq.txt'
    elif len(suggSeq)>0:
        seqFile = suggSeq[0]

    # if no sequence stars, download from PS1...
    if len(suggSeq)==0:
        if hasPS1 == True:
            print('Could not find sequence stars in this filter, but have PS1 for coordinates')
        else:
            print('No sequence star data found locally...')
            try:
                PS1catalog(RAdec[0],RAdec[1])
                seqFile = 'PS1_seq.txt'
                print('Found PS1 stars')
                trySDSS = False
            except:
                print('PS1 query failed')
                trySDSS = True
        # if not in PS1 or filter is u-band, query also SDSS
        if f == 'u' or trySDSS == True:
            try:
                SDSScatalog(RAdec[0],RAdec[1])
                seqFile = 'SDSS_seq.txt'
                print('Found SDSS stars')
            except:
                print('SDSS query failed')


    print('\n####################\n\nSequence star magnitudes: '+seqFile)


    seqDat = np.genfromtxt(seqFile)
    seqHead = np.genfromtxt(seqFile,skip_footer=len(seqDat)-1,dtype=str)

    seqMags = {}
    for i in range(len(seqHead)-2):
        seqMags[seqHead[i+2]] = seqDat[:,i+2]


    already_clean = False

    zero_shift_image = ''
    
    dostack = 'n'
    
    ims2 = ims1[:,0].copy()

    if stack==True:
        dostack = 'y'
        if len(ims1[:,0]) < 2:
            print('\nOnly 1 image in filter, skipping stacking!')
            dostack = 'n'
        elif len(glob.glob('stack*_'+f+'.fits')) > 0:
            print('Already exists:')
            print(glob.glob('stack*_'+f+'.fits'))
            if not quiet:
                use_existing = input('\nStack(s) already found in filter, use existing? \n (Note this will ignore unstacked data!) [y]')
                if not use_existing: use_existing = 'y'
                if use_existing == 'y':
                    dostack = 'n'
                    ims2 = glob.glob('stack*_'+f+'.fits')
            else:
                dostack = 'n'


        if dostack in ('y','yes'):
            ims2 = []
            print('\nAligning and stacking images...')
            mintime = np.min(filtertab[:,2][filtertab[:,1]==f].astype(float))
            maxtime = np.max(filtertab[:,2][filtertab[:,1]==f].astype(float))
            timerange = maxtime-mintime
            tlim = mintime
            binedges = [mintime]
            while tlim < maxtime:
                tlim += timebins
                binedges.append(tlim)
            print('Time bins:')
            print(binedges)

            for bin in range(len(binedges)-1):
                time_in_range = (ims1[:,2].astype(float)>=binedges[bin])&(ims1[:,2].astype(float)<binedges[bin+1])
                stacktab = ims1[time_in_range]
                if len(stacktab) > 1:
                    try:
                        mjdstack = np.mean(stacktab[:,2].astype(float))

                        zero_shift_image = stacktab[:,0][0]
                        
            #            imshifts = {} # dictionary to hold the x and y shift pairs for each image
            #            for im1 in ims1[:,0]:
            #                ## phase_cross_correlation is a function that calculates shifts by comparing 2-D arrays
            #                imshift, imshifterr, diffphase = phase_cross_correlation(
            #                    fits.getdata(zero_shift_image),
            #                    fits.getdata(im1))
            #                imshifts[im1] = imshift
            #
            #            ## new dictionary for shifted image data:
            #            shifted_data = {}
            #            for im1 in imshifts:
            #                shifted_data[im1] = interp.shift(
            #                    fits.getdata(im1),
            #                    imshifts[im1])
            #                shifted_data[im1][shifted_data[im1] == 0] = 'nan'

                        if clean == True:
                            clean_zero, mask = lacosmic(fits.getdata(zero_shift_image))
                            already_clean = True

                        shifted_data = {}
                        for im1 in stacktab[:,0]:
                            if clean == True:
                                print('Cleaning cosmics')
                                clean_source, mask = lacosmic(fits.getdata(im1))
                                registered, footprint = aa.register(clean_source, clean_zero, fill_value=np.nan)
                            else:
                                registered, footprint = aa.register(fits.getdata(im1), fits.getdata(zero_shift_image), fill_value=np.nan)

                            shifted_data[im1] = registered

                        shifted_data_cube = np.stack([shifted_data[im1] for im1 in stacktab[:,0]])
                        stacked_data = np.nanmedian(shifted_data_cube, axis=0)
                        
                        stackheader = fits.getheader(zero_shift_image)
                        
                        stackheader['MJD'] = mjdstack
                        stackheader['MJD-OBS'] = mjdstack

                        fits.writeto('stack_'+str(np.round(mjdstack,2))+'_'+f+'.fits',
                                stacked_data,header=stackheader,
                                overwrite=True)
                        
                        ims2.append('stack_'+str(np.round(mjdstack,2))+'_'+f+'.fits')
                        
                        print('Stack done: ' + str(np.round(mjdstack,2)))
                        
                        if sub == True and has_template == True:
                            filtertab2.append(['stack_'+str(np.round(mjdstack,2))+'_'+f+'.fits', filtername, mjdstack, template])
                        else:
                            filtertab2.append(['stack_'+str(np.round(mjdstack,2))+'_'+f+'.fits', filtername, mjdstack, ''])

                    except:
                        print('Alignment/stacking failed in bin, using single images')
                        for single_im in stacktab[:,0]:
                            ims2.append(single_im)
                            already_clean = False
                            
                elif len(stacktab[:,0]) == 1:
                    print('Only one image in bin')
                    ims2.append(stacktab[0][0])
                    already_clean = False
                    
                else:
                    pass
                
    
#################################
# Part four: do some photometry
#################################

    counter = 1

    for image in ims2:
    
        try:
                
            # PSF parameters that can be changed each time
            stamprad = stamprad0
            psfthresh = psfthresh0
            samp = samp0
        
            plt.clf()

            print('\n> Image: '+image+'  (number %d of %d in filter)' %(counter,len(ims2)))
            
            counter += 1

            try:
                mjd = filtertab[:,2][filtertab[:,0]==image][0]
            except:
                try:
                    mjd = filtertab2[:,2][filtertab2[:,0]==image][0]
                except:
                    mjd = fits.getval(image,keyword='MJD')


            mjd = float(mjd)

            comment1 = ''

            try:
                target_name = fits.getval(image,'OBJECT')
            except:
                try:
                    target_name = fits.getval(image,'UNKNOWN')
                except:
                    target_name = 'UNKNOWN'
                    

    ### NEED TO MAKE COMPATIBLE WITH NAMES OF IMAGE IN SUBTRACTION PART

            im = fits.open(image)

            try:
                im[0].verify('fix')

                data = im[0].data
                header = im[0].header
                checkdat = len(data)

            except:
                im[1].verify('fix')

                data = im[1].data
                header = im[1].header
                
            try:
                gain = header['GAIN']
            except:
                gain = 1.0

            if clean == True and already_clean == False:
                print('\nCleaning cosmics...')
                data, cosmicmask = lacosmic(data)
                print('Done')

            # Set up sequence stars, initial steps
            
            mag_range = (seqMags[f]>magmax)&(seqMags[f]<magmin)
            
            co = astropy.wcs.WCS(header=header).all_world2pix(seqDat[:,0],seqDat[:,1],1)
            co = np.array(list(zip(co[0],co[1])))
            
            co = co[mag_range]

            # Remove any stars falling outside the image or too close to edge
            inframe = (co[:,0]>stamprad)&(co[:,0]<len(data[0])-stamprad)&(co[:,1]>stamprad)&(co[:,1]<len(data)-stamprad)
            co = co[inframe]

            # Find and remove bad pixels and sequence stars: nans, regions of zero etc.
            goodpix = []
            for c in range(len(co)):
                if data[int(co[c][1]),int(co[c][0])] != 0:  # Remember RA, dec = y, x in our data array
                    goodpix.append(c)

            co = co[goodpix]

            orig_co = co.copy()


            # background subtraction:

            # first clean bad regions before fitting
            data[data==0] = np.median(data)

            data[np.isnan(data)] = np.median(data)

            data[np.isinf(data)] = np.median(data)

            print('\nSubtracting background...')

            bkg = photutils.background.Background2D(data,box_size=bkgbox)
            
            bkg_error = bkg.background_rms

            data = data.astype(float) - bkg.background
            
            try:
                err_array = calc_total_error(data, bkg_error, gain)
            except:
                data_adu = data.copy()
                data = np.array([i.astype(float) for i in data_adu])
                bkg_error_adu = bkg_error.copy()
                bkg_error = np.array([i.astype(float) for i in bkg_error_adu])
                err_array = calc_total_error(data, bkg_error, gain)

            axBKG = plt.subplot2grid((2,5),(0,2))
            
            axBKG.imshow(bkg.background, origin='lower',cmap='viridis')
            
            axBKG.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

            axBKG.set_title('Background')
        
            print('Done')

        ########## plot data

            plt.subplots_adjust(left=0.05,right=0.99,top=0.99,bottom=-0.05)


            ax1 = plt.subplot2grid((2,5),(0,0),colspan=2,rowspan=2)


            ax1.imshow(data, origin='lower',cmap='gray',
                        vmin=visualization.ZScaleInterval().get_limits(data)[0],
                        vmax=visualization.ZScaleInterval().get_limits(data)[1])

            ax1.set_title(image+' ('+f+')')

            ax1.set_xlim(0,len(data))
            ax1.set_ylim(0,len(data))

            ax1.get_yaxis().set_visible(False)
            ax1.get_xaxis().set_visible(False)

            print('\nFinding sequence star centroids...')

            # Mark sequence stars
            ax1.errorbar(co[:,0],co[:,1],fmt='s',mfc='none',markeredgecolor='C0',
                            markersize=8,markeredgewidth=1.5)


            SNco = np.array(astropy.wcs.WCS(header=header).all_world2pix(RAdec[0],RAdec[1],0))

            ax1.errorbar(SNco[0],SNco[1],fmt='o',markeredgecolor='r',mfc='none',
                            markeredgewidth=3,markersize=20)

            print('Done')

            plt.draw()

            plt.tight_layout(pad=0.5)

            # Manual shifts:
            if shifts:
                x_sh = input('\n> Add approx pixel shift in x? ['+str(x_sh_1)+']  ')
                if not x_sh: x_sh = x_sh_1
                x_sh = int(x_sh)
                x_sh_1 = x_sh

                y_sh = input('\n> Add approx pixel shift in y? ['+str(y_sh_1)+']  ')
                if not y_sh: y_sh = y_sh_1
                y_sh = int(y_sh)
                y_sh_1 = y_sh

                co[:,0] += x_sh
                co[:,1] += y_sh

            # And centroid
            co[:,0],co[:,1] = photutils.centroids.centroid_sources(data,co[:,0],co[:,1],
                                        centroid_func=photutils.centroids.centroid_2dg)

            del_x = np.nanmedian(co[:,0]-orig_co[:,0])
            del_y = np.nanmedian(co[:,1]-orig_co[:,1])
            sig_x = np.nanstd(co[:,0]-orig_co[:,0])
            sig_y = np.nanstd(co[:,1]-orig_co[:,1])
            
            SNco[0] += del_x
            SNco[1] += del_y


            found = (abs(co[:,0]-orig_co[:,0])<max(abs(del_x)*10,5))&(abs(co[:,1]-orig_co[:,1])<max(abs(del_y)*10,5))

            co = co[found]

            for j in range(len(co)):
                apcircle = Circle((co[j,0], co[j,1]), aprad, facecolor='none',
                        edgecolor='b', linewidth=1, alpha=1)
                ax1.add_patch(apcircle)

                ax1.text(co[j,0]+20,co[j,1]-20,str(j+1),color='k',fontsize=14)



            # Define apertures and do simple photometry on sequence stars

            print('\nDoing aperture photometry...')

            photaps = photutils.CircularAperture(co, r=aprad)

            photTab = photutils.aperture_photometry(data, photaps, error=err_array)

            print('Done')


            # PSF Photometry
            print('\nBuilding PSF...')

            # Required formats for photutils:
            nddata = astropy.nddata.NDData(data=data)
            psfinput = astropy.table.Table()
            psfinput['x'] = co[:,0]
            psfinput['y'] = co[:,1]


            # Create model from sequence stars
            happy = 'n'
            while happy not in ('y','yes'):
            
                # psf quality tests
                sumpsf = -1
                minpsf = -1
                x_peak = 0
                y_peak = 0
                
                psf_iter = 0
                
                while (sumpsf < 0) or (minpsf < -0.01) or ((x_peak/len(psf) < 0.4) or (x_peak/len(psf) > 0.6)) or ((y_peak/len(psf) < 0.4) or (y_peak/len(psf) > 0.6)) and (psf_iter < 5):
                
                    if psf_iter > 0:
                        print('PSF failed quality checked, randomly varying parameters and trying again')
                        stamprad += np.random.randint(10)-5
                        psfthresh += np.random.randint(10)

                    # extract stars from image
                    psfstars = photutils.psf.extract_stars(nddata, psfinput[photTab['aperture_sum']>psfthresh*photTab['aperture_sum_err']], size=2*stamprad+5)
                    while(len(psfstars))<5 and psfthresh>0:
                        print('Warning: too few PSF stars with threshold '+str(psfthresh)+' sigma, trying lower sigma')
                        psfthresh -= 1
                        psfstars = photutils.psf.extract_stars(nddata, psfinput[photTab['aperture_sum']>psfthresh*photTab['aperture_sum_err']], size=2*stamprad+5)
                    if len(psfstars)<5:
                        psfthresh = psfthresh0
                        print('Could not find 5 PSF stars, trying for 3...')
                        while(len(psfstars))<3 and psfthresh>0:
                            psfthresh -= 1
                            psfstars = photutils.psf.extract_stars(nddata, psfinput[photTab['aperture_sum']>psfthresh*photTab['aperture_sum_err']], size=2*stamprad+5)
                        if psfthresh < 3:
                            empirical = False
                            break

                    ax1.clear()
                    
                    ax1.imshow(data, origin='lower',cmap='gray',
                                vmin=visualization.ZScaleInterval().get_limits(data)[0],
                                vmax=visualization.ZScaleInterval().get_limits(data)[1])

                    ax1.set_title(image+' ('+f+')')

                    ax1.set_xlim(0,len(data))
                    ax1.set_ylim(0,len(data))

                    ax1.get_yaxis().set_visible(False)
                    ax1.get_xaxis().set_visible(False)
                    
                    ax1.errorbar(SNco[0],SNco[1],fmt='o',markeredgecolor='r',mfc='none',
                                    markeredgewidth=3,markersize=20)

                    ax1.errorbar(co[:,0],co[:,1],fmt='s',mfc='none',markeredgecolor='C0',
                                                    markersize=8,markeredgewidth=1.5)

                    ax1.errorbar(psfstars.center_flat[:,0],psfstars.center_flat[:,1],fmt='*',mfc='none', markeredgecolor='lime',markeredgewidth=2, markersize=20,label='Used in PSF fit')

                    ax1.legend(frameon=True,fontsize=16,loc='upper left')


                    # build PSF
                
                    try:
                        epsf_builder = photutils.EPSFBuilder(maxiters=10,recentering_maxiters=5,
                                        oversampling=samp,smoothing_kernel='quadratic',shape=2*stamprad-1)
                        epsf, fitted_stars = epsf_builder(psfstars)

                        psf = epsf.data
                        
                        x_peak = np.where(psf==psf.max())[1][0]
                        y_peak = np.where(psf==psf.max())[0][0]
                        
                        minpsf = np.min(psf)
                        sumpsf = np.sum(psf)
                        
                        psf_iter += 1
      
                        ax2 = plt.subplot2grid((2,5),(0,3))

                        ax2.imshow(psf, origin='lower',cmap='gray',
                                    vmin=visualization.ZScaleInterval().get_limits(psf)[0],
                                    vmax=visualization.ZScaleInterval().get_limits(psf)[1])

                        ax2.get_yaxis().set_visible(False)
                        ax2.get_xaxis().set_visible(False)

                        ax2.set_title('PSF')

                        plt.draw()


                        ax3 = plt.subplot2grid((2,5),(0,4),projection='3d')

                        tmpArr = range(len(psf))

                        X, Y = np.meshgrid(tmpArr,tmpArr)

                        ax3.plot_surface(X,Y,psf,rstride=1,cstride=1,cmap='viridis_r',alpha=0.5)

                        ax3.set_zlim(np.min(psf),np.max(psf)*1.1)

                        ax3.set_axis_off()

                        plt.draw()

                        plt.tight_layout(pad=0.5)
                        
                        empirical = True

                                
                    except:
                        print('PSF fit failed (usually a weird EPSF_builder error):\nTrying different stamp size usually fixes!')
                        empirical = False

                        psf_iter += 1


                if not quiet:
                    if empirical == True:
                        happy = input('\nProceed with this PSF? [y] ')
                        if not happy: happy = 'y'
                    else:
                        happy = input('\nProceed with simple Gaussian (y) or try varying parameters (n)? [n] ')
                        if not happy: happy = 'n'
                    if happy not in ('y','yes'):
                        stamprad1 = input('Try new cutout radius: [' +str(stamprad)+'] ')
                        if not stamprad1: stamprad1 = stamprad
                        stamprad = int(stamprad1)

                        psfthresh1 = input('Try new inclusion threshold: [' +str(psfthresh)+' sigma] ')
                        if not psfthresh1: psfthresh1 = psfthresh
                        psfthresh = int(psfthresh1)
                        
                        samp1 = input('Try new PSF oversampling: [' +str(samp)+'] ')
                        if not samp1: samp1 = samp
                        samp = int(samp1)
                else:
                    happy = 'y'
                    
                    
            if empirical == False:
                print('\nNo PSF determined, using basic Gaussian model')
                
                if not quiet:
                    sigma_gauss = input('Please specify width (sigma) in pixels ['+str(sigma_gauss_0)+'] ')
                    if not sigma_gauss: sigma_gauss = sigma_gauss_0
                else:
                    sigma_gauss = sigma_gauss_0
                sigma_gauss = float(sigma_gauss)
                
                epsf = IntegratedGaussianPRF(sigma=sigma_gauss)
                
                psf = np.zeros((2*stamprad+1,2*stamprad+1))
                
                for xt in np.arange(2*stamprad+1):
                    for yt in np.arange(2*stamprad+1):
                        psf[xt,yt] = epsf.evaluate(xt,yt,x_0=stamprad,y_0=stamprad,sigma=sigma_gauss,flux=1)
            

                ax2 = plt.subplot2grid((2,5),(0,3))

                ax2.imshow(psf, origin='lower',cmap='gray',
                            vmin=visualization.ZScaleInterval().get_limits(psf)[0],
                            vmax=visualization.ZScaleInterval().get_limits(psf)[1])

                ax2.get_yaxis().set_visible(False)
                ax2.get_xaxis().set_visible(False)

                ax2.set_title('PSF')

                plt.draw()


                ax3 = plt.subplot2grid((2,5),(0,4),projection='3d')

                tmpArr = range(len(psf))

                X, Y = np.meshgrid(tmpArr,tmpArr)

                ax3.plot_surface(X,Y,psf,rstride=1,cstride=1,cmap='viridis_r',alpha=0.5)

                ax3.set_zlim(np.min(psf),np.max(psf)*1.1)

                ax3.set_axis_off()

                plt.draw()

                plt.tight_layout(pad=0.5)

                        
                    
            scipsf = fits.PrimaryHDU(psf)
            scipsf.writeto('sci_psf.fits',overwrite=True)


            # determine aperture size and correction
            
    #            pix_frac = 0.5 # Optimal radius for S/N is R ~ FWHM. But leads to large aperture correction
    #            pix_frac = 0.1 # Aperture containing 90% of flux. Typically gives a radius ~ 2*FWHM

            pix_frac = np.max([1-apfrac,0.05])

            aprad_opt = np.sqrt(len(psf[psf>np.max(psf)*pix_frac])/np.pi)
            
            test_ap = photutils.CircularAperture([len(psf[0])/2,len(psf[0])/2], r=aprad_opt)
            
            testTab = photutils.aperture_photometry(psf,test_ap)
            
            apfrac_verify = testTab['aperture_sum'][0]/np.sum(psf) # fraction of flux contained in aprad_opt

            ap_corr = 2.5*np.log10(apfrac_verify)


            print('\n\nDoing aperture photometry for optimal aperture...')

            photaps_opt = photutils.CircularAperture(co, r=aprad_opt)

            photTab_opt = photutils.aperture_photometry(data, photaps_opt, error=err_array)

            detected = (photTab_opt['aperture_sum']>3*photTab_opt['aperture_sum_err'])

            faintest_object = max(seqMags[f][mag_range][inframe][goodpix][found][detected])


            print('Done')



            print('\nStarting PSF photometry...')

            psfcoordTable = astropy.table.Table()

            psfcoordTable['x_0'] = co[:,0]
            psfcoordTable['y_0'] = co[:,1]
            psfcoordTable['flux_0'] = photTab['aperture_sum']

            grouper = photutils.psf.DAOGroup(crit_separation=stamprad)

            # need an odd number of pixels to fit PSF
            fitrad = 2*stamprad + 1

            psfphot = photutils.psf.BasicPSFPhotometry(group_maker=grouper,
                            bkg_estimator=photutils.background.MMMBackground(),
                            psf_model=epsf, fitshape=fitrad,
                            finder=None, aperture_radius=min(stamprad,2*aprad_opt))

            psfphotTab = psfphot.do_photometry(data, init_guesses=psfcoordTable)

            psfsubIm = psfphot.get_residual_image()

            ax1.imshow(psfsubIm, origin='lower',cmap='gray',
                        vmin=visualization.ZScaleInterval().get_limits(data)[0],
                        vmax=visualization.ZScaleInterval().get_limits(data)[1])

            goodStars = (photTab['aperture_sum']>5*photTab['aperture_sum_err'])

            ax1.errorbar(co[:,0][goodStars],co[:,1][goodStars], fmt='s',mfc='none', markeredgecolor='midnightblue',markersize=8,markeredgewidth=2.5,label='SNR>5')


            ax1.legend(frameon=True,fontsize=16,loc='upper left')


            print('Done')

        ########## Zero point from seq stars

            print('\nComputing image zeropoint...')

            if f in seqMags:
            
                magmax2 = magmax
                magmin2 = magmin
                
                
                happy = 'n'
                while happy not in ('y','yes'):

                    if magmin2 < magmax2:
                        print('error: magmin brighter than magmax - resetting to defaults')
                        magmin2 = 22
                        magmax2 = 16.5

                    mag_range_2 = (seqMags[f][mag_range][inframe][goodpix][found][goodStars]>=magmax2) & (seqMags[f][mag_range][inframe][goodpix][found][goodStars]<=magmin2)
                
                    
                    # PSF zeropoint
                    flux = np.array(psfphotTab['flux_fit'])[goodStars]
                    
                    seqIm = -2.5*np.log10(flux)

                    zpList1 = seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2] - seqIm[mag_range_2]

                    axZP = plt.subplot2grid((2,5),(1,2))

                    axZP.scatter(seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2], zpList1,color='r')

                    zp1 = np.nanmedian(zpList1)
                    errzp1 = np.nanstd(zpList1)

                    print('\nInitial zeropoint =  %.2f +/- %.2f\n' %(zp1,errzp1))
                    print('Checking for bad stars...')

                    if len(seqIm) > 10:
                        checkMags = np.abs(seqIm[mag_range_2]+zp1 - seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2]) < errzp1*sigClip
                    else:
                        checkMags = np.abs(seqIm[mag_range_2]+zp1 - seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2]) < 0.5

    #                if False in checkMags:
    #                    print('Rejecting stars from ZP: ')
    #                    print(psfphotTab['id'][goodStars][mag_range_2][~checkMags])
    #                    print('Bad ZPs:')
    #                    print( seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][~checkMags] - seqIm[mag_range_2][~checkMags])

                    ax1.errorbar(co[:,0][goodStars][mag_range_2][~checkMags],
                                co[:,1][goodStars][mag_range_2][~checkMags],fmt='x',mfc='none',
                                markeredgewidth=2, color='C3',
                                markersize=8,label='Sigma clipped from ZP')

                    ax1.legend(frameon=True,fontsize=16,loc='upper left')

                    zpList = seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][checkMags] - seqIm[mag_range_2][checkMags]

                    ZP = np.median(zpList)
                    errZP = np.std(zpList)#/np.sqrt(len(zpList))

                    ZP_psf = np.median(zpList)
                    errZP_psf = np.std(zpList)#/np.sqrt(len(zpList))

                    axZP.scatter(seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][checkMags], zpList,color='k',label='psf')

                    axZP.axhline(ZP,linestyle='-',color='k')
                    axZP.axhline(ZP-errZP,linestyle='--',color='k')
                    axZP.axhline(ZP+errZP,linestyle='--',color='k')

                    axZP.set_xlabel('Magnitude')
                    axZP.set_title('Zero point')
                    
                    axZP.set_ylim(max(max(zpList1)+0.5,ZP+1),min(min(zpList1)-0.5,ZP-1))
                    axZP.set_xlim(magmax2,magmin2)
                    
                    plt.draw()
     
                    # optimal aperture ZP
                    flux = np.array(photTab_opt['aperture_sum'])[goodStars]
                    
                    seqIm = -2.5*np.log10(flux)

                    zpList1 = seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2] - seqIm[mag_range_2]

                    zp1 = np.nanmedian(zpList1)
                    errzp1 = np.nanstd(zpList1)

                    if len(seqIm) > 10:
                        checkMags = np.abs(seqIm[mag_range_2]+zp1 - seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2]) < errzp1*sigClip
                    else:
                        checkMags = np.abs(seqIm[mag_range_2]+zp1 - seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2]) < 0.5

        #                if False in checkMags:
        #                    print('Rejecting stars from ZP: ')
        #                    print(psfphotTab['id'][goodStars][mag_range_2][~checkMags])
        #                    print('Bad ZPs:')
        #                    print( seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][~checkMags] - seqIm[mag_range_2][~checkMags])


                    zpList = seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][checkMags] - seqIm[mag_range_2][checkMags]

                    ZP_opt = np.median(zpList)
                    errZP_opt = np.std(zpList)#/np.sqrt(len(zpList))


                    axZP.scatter(seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][checkMags], zpList,color='C0',marker='x',label='aperture')

                    axZP.axhline(ZP_opt,linestyle='-',color='C0')
                    axZP.axhline(ZP_opt-errZP_opt,linestyle='--',color='C0')
                    axZP.axhline(ZP_opt+errZP_opt,linestyle='--',color='C0')

                    axZP.legend(loc='lower left',fontsize=16,frameon=True, ncol=2,columnspacing=0.6,handletextpad=-0.2)

                    # big aperture ZP
                    flux = np.array(photTab['aperture_sum'])[goodStars]
                    
                    seqIm = -2.5*np.log10(flux)

                    zpList1 = seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2] - seqIm[mag_range_2]

                    zp1 = np.nanmedian(zpList1)
                    errzp1 = np.nanstd(zpList1)

                    if len(seqIm) > 10:
                        checkMags = np.abs(seqIm[mag_range_2]+zp1 - seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2]) < errzp1*sigClip
                    else:
                        checkMags = np.abs(seqIm[mag_range_2]+zp1 - seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2]) < 0.5

        #                if False in checkMags:
        #                    print('Rejecting stars from ZP: ')
        #                    print(psfphotTab['id'][goodStars][mag_range_2][~checkMags])
        #                    print('Bad ZPs:')
        #                    print( seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][~checkMags] - seqIm[mag_range_2][~checkMags])


                    zpList = seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][checkMags] - seqIm[mag_range_2][checkMags]

                    ZP_ap = np.median(zpList)
                    errZP_ap = np.std(zpList)#/np.sqrt(len(zpList))


                    print('\nPSF Zeropoint = %.2f +/- %.2f' %(ZP_psf,errZP_psf))
                    print('Big aperture Zeropoint = %.2f +/- %.2f' %(ZP_ap,errZP_ap))
                    print('Optimal aperture Zeropoint = %.2f +/- %.2f' %(ZP_opt,errZP_opt))

                    
                    if not quiet:
                        happy = input('\nProceed with this zeropoint? [y] ')
                        if not happy: happy = 'y'
                        if happy not in ('y','yes'):
                            magmax1 = input('Use new maximum mag: [' +str(magmax2)+'] ')
                            if not magmax1: magmax1 = magmax2
                            magmax2 = float(magmax1)
                                                    
                            magmin1 = input('Use new minimum mag: [' +str(magmin2)+'] ')
                            if not magmin1: magmin1 = magmin2
                            magmin2 = float(magmin1)
                    else:
                        happy = 'y'


            else:
                ZP = np.nan
                errZP = np.nan
                ZP_psf = np.nan
                errZP_psf = np.nan
                ZP_opt = np.nan
                errZP_opt = np.nan
                ZP_ap = np.nan
                errZP_ap = np.nan

                print('\nCould not determine ZP (no sequence star mags in filter?) : instrumental mag only!!!')

            
        ########### Template subtraction

            if sub == True and has_template == False:
                print('\nNo template associated to image, skipping subtraction')

            elif sub == True and has_template == True:
      
                print('\nAligning template image and building template PSF')

                if noalign == True:
                    im2 = fits.open(template)
                    
                else:
                    tmp = fits.open(template)

                    try:
                        tmp[0].verify('fix')

                        data2 = tmp[0].data
                        header2 = tmp[0].header
                        checkdat2 = len(data2)
                        
                        tmp = tmp[0]

                    except:
                        tmp[1].verify('fix')

                        data2 = tmp[1].data
                        header2 = tmp[1].header
                        
                        tmp = tmp[1]




                    ### Using Astroalign

                    try:
                        im_fixed = np.array(data, dtype="<f4")
                        tmp_fixed = np.array(data2, dtype="<f4")

                        registered, footprint = aa.register(tmp_fixed, im_fixed)

                        tmp_masked = np.ma.masked_array(registered, footprint, fill_value=np.nanmedian(tmp_fixed)).filled()

                        tmp_masked[np.isnan(tmp_masked)] = np.nanmedian(tmp_fixed)

                        hdu2 = fits.PrimaryHDU(tmp_masked)

                        hdu2.writeto('tmpl_aligned.fits',overwrite=True)
            
                    except:
                    
                        print('Astroalign failed to match images, trying Reproject')
                    
                    ### Using Reproject

                        tmp_resampled, footprint = reproject_interp(tmp, header)

                        tmp_resampled[np.isnan(tmp_resampled)] = np.nanmedian(tmp_resampled)

                        hdu2 = fits.PrimaryHDU(tmp_resampled)

                        hdu2.writeto('tmpl_aligned.fits',overwrite=True)


                    im2 = fits.open('tmpl_aligned.fits')


                try:
                    im2[0].verify('fix')

                    data2 = im2[0].data
                    header2 = im2[0].header
                    checkdat2 = len(data2)

                except:
                    im2[1].verify('fix')

                    data2 = im2[1].data
                    header2 = im2[1].header
                    
                try:
                    gain2 = header2['GAIN']
                except:
                    gain2 = 1.0

                bkg2 = photutils.background.Background2D(data2,box_size=bkgbox)

                data2 = data2.astype(float) - bkg2.background
                
                bkg_error2 = bkg2.background_rms

                try:
                    err_array2 = calc_total_error(data2, bkg_error2, gain2)
                except:
                    data2_adu = data2.copy()
                    data2 = np.array([i.astype(float) for i in data2_adu])
                    bkg_error2_adu = bkg_error2.copy()
                    bkg_error2 = np.array([i.astype(float) for i in bkg_error2_adu])
                    err_array2 = calc_total_error(data2, bkg_error2, gain2)

                
                fig2 = plt.figure(2,(14,7))
                plt.clf()
                plt.ion()
                plt.show()

                plt.subplots_adjust(left=0.05,right=0.99,top=0.99,bottom=-0.05)


                ax1t = plt.subplot2grid((2,5),(0,0),colspan=2,rowspan=2)


                ax1t.imshow(data2, origin='lower',cmap='gray',
                            vmin=visualization.ZScaleInterval().get_limits(data2)[0],
                            vmax=visualization.ZScaleInterval().get_limits(data2)[1])

                ax1t.set_title(template)

                ax1t.set_xlim(0,len(data2))
                ax1t.set_ylim(0,len(data2))

                ax1t.get_yaxis().set_visible(False)
                ax1t.get_xaxis().set_visible(False)

                axBKGt = plt.subplot2grid((2,5),(0,2))

                axBKGt.imshow(bkg2.background, origin='lower',cmap='viridis')

                axBKGt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

                axBKGt.set_title('Background')


                co2 = co.copy()

                co2[:,0],co2[:,1] = photutils.centroids.centroid_sources(data2,co2[:,0],co2[:,1],
                                            centroid_func=photutils.centroids.centroid_2dg)

                psfinput2 = astropy.table.Table()
                psfinput2['x'] = co2[:,0]
                psfinput2['y'] = co2[:,1]



                nddata2 = astropy.nddata.NDData(data=data2)


                photaps2 = photutils.CircularAperture(co2, r=aprad)

                photTab2 = photutils.aperture_photometry(data2, photaps2, err_array2)


                print('\nBuilding template image PSF for subtraction')

                stamprad2 = stamprad
                psfthresh2 = psfthresh0
                samp2 = samp

                # Create model from sequence stars
                happy = 'n'
                while happy not in ('y','yes'):

                    sumpsf = -1
                    minpsf = -1
                    x_peak = 0
                    y_peak = 0
                    
                    psf_iter = 0
                    
                    while (sumpsf < 0) or (minpsf < -0.01) or ((x_peak/len(psf2) < 0.4) or (x_peak/len(psf2) > 0.6)) or ((y_peak/len(psf2) < 0.4) or (y_peak/len(psf2) > 0.6)) and (psf_iter < 5):
                    
                        if psf_iter > 0:
                            print('PSF failed quality checked, randomly varying parameters and trying again')
                            stamprad2 += np.random.randint(10)-5
                            psfthresh2 = psfthresh0 + np.random.randint(20)

                        psfstars2 = photutils.psf.extract_stars(nddata2, psfinput2[photTab2['aperture_sum']>psfthresh2*photTab2['aperture_sum_err']], size=2*stamprad2+5)
                        while(len(psfstars2))<5 and psfthresh2>0:
                            print('Warning: too few PSF stars with threshold '+str(psfthresh2)+' sigma, trying lower sigma')
                            psfthresh2 -= 1
                            psfstars2 = photutils.psf.extract_stars(nddata2, psfinput2[photTab2['aperture_sum']>psfthresh2*photTab2['aperture_sum_err']], size=2*stamprad2+5)
                        if len(psfstars2)<5:
                            print('Could not find 5 PSF stars, trying for 3...')
                            psfthresh2 = psfthresh0
                            while(len(psfstars2))<3 and psfthresh2>0:
                                psfthresh2 -= 1
                                psfstars2 = photutils.psf.extract_stars(nddata2, psfinput2[photTab2['aperture_sum']>psfthresh2*photTab2['aperture_sum_err']], size=2*stamprad2+5)
                            if psfthresh2 < 3:
                                empirical = False
                                break


                        ax1t.errorbar(psfstars2.center_flat[:,0],psfstars2.center_flat[:,1],fmt='*',mfc='none', markeredgecolor='lime',markeredgewidth=2, markersize=20,label='Used in PSF fit')
                        
                        
                        # build PSF
                        try:
                            epsf_builder = photutils.EPSFBuilder(maxiters=10,recentering_maxiters=5,
                                            oversampling=samp2,smoothing_kernel='quadratic',shape=2*stamprad2-1)
                            epsf2, fitted_stars2 = epsf_builder(psfstars2)

                            psf2 = epsf2.data
                            
                            x_peak = np.where(psf2==psf2.max())[1][0]
                            y_peak = np.where(psf2==psf2.max())[0][0]
                            
                            minpsf = np.min(psf2)
                            sumpsf = np.sum(psf2)
                            
                            psf_iter += 1


                            ax2t = plt.subplot2grid((2,5),(0,3))

                            ax2t.imshow(psf2, origin='lower',cmap='gray',
                                        vmin=visualization.ZScaleInterval().get_limits(psf2)[0],
                                        vmax=visualization.ZScaleInterval().get_limits(psf2)[1])

                            ax2t.get_yaxis().set_visible(False)
                            ax2t.get_xaxis().set_visible(False)

                            ax2t.set_title('PSF')

                            plt.draw()

                            

                            ax3t = plt.subplot2grid((2,5),(0,4),projection='3d')

                            tmpArr2 = range(len(psf2))

                            X2, Y2 = np.meshgrid(tmpArr2,tmpArr2)

                            ax3t.plot_surface(X2,Y2,psf2,rstride=1,cstride=1,cmap='viridis_r',alpha=0.5)

                            ax3t.set_zlim(np.min(psf2),np.max(psf2)*1.1)

                            ax3t.set_axis_off()

                            plt.draw()

                            plt.tight_layout(pad=0.5)
                            
                            empirical = True

         
                        except:
                            print('PSF fit failed (usually a weird EPSF_builder error):\nTrying different stamp size usually fixes!')
                            empirical = False

                            psf_iter += 1


                    if not quiet:
                        if empirical == True:
                            happy = input('\nProceed with this template PSF? [y] ')
                            if not happy: happy = 'y'
                        else:
                            happy = input('\nProceed with simple Gaussian (y) or try varying parameters (n)? [n] ')
                            if not happy: happy = 'n'
                        if happy not in ('y','yes'):
                            stamprad1 = input('Try new cutout radius: [' +str(stamprad2)+']')
                            if not stamprad1: stamprad1 = stamprad2
                            stamprad2 = int(stamprad1)

                            psfthresh1 = input('Try new inclusion threshold: [' +str(psfthresh2)+' sigma]')
                            if not psfthresh1: psfthresh1 = psfthresh2
                            psfthresh2 = int(psfthresh1)
                            
                            samp1 = input('Try new PSF oversampling: [' +str(samp2)+']')
                            if not samp1: samp1 = samp2
                            samp2 = int(samp1)
                    else:
                        happy = 'y'
                        
                
                if empirical == False:
                    print('\nNo PSF determined, using basic Gaussian model')
                    
                    if not quiet:
                        sigma_gauss2 = input('Please specify width (sigma) in pixels ['+str(sigma_gauss_0)+'] ')
                        if not sigma_gauss2: sigma_gauss2 = sigma_gauss_0
                    else:
                        sigma_gauss2 = sigma_gauss_0
                    sigma_gauss2 = float(sigma_gauss2)
                    
                    epsf2 = IntegratedGaussianPRF(sigma=sigma_gauss2)
                    
                    psf2 = np.zeros((2*stamprad2+1,2*stamprad2+1))
                    
                    for xt in np.arange(2*stamprad2+1):
                        for yt in np.arange(2*stamprad2+1):
                            psf2[xt,yt] = epsf2.evaluate(xt,yt,x_0=stamprad2,y_0=stamprad2,sigma=sigma_gauss2,flux=1)



                tmppsf = fits.PrimaryHDU(psf2)
                tmppsf.writeto('tmpl_psf.fits',overwrite=True)

                
                plt.close(fig2)
                
                            
                # Make cutouts for subtraction

                cutout_loop = 'y'
                
                
                cutoutsize_new = cutoutsize
                if f == 'u' and cutoutsize < 1600:
                    cutoutsize_new = 1600
                sci_sat_new = sci_sat
                tmpl_sat_new = tmpl_sat

                data_orig = np.copy(data)
                header_orig = header.copy()

                im_sci = fits.PrimaryHDU()
                im_sci.data = data_orig
                im_sci.header = header_orig
                
                
                im2 = fits.open('tmpl_aligned.fits')

                try:
                    im2[0].verify('fix')

                    data2 = im2[0].data
                    checkdat2 = len(data2)

                    im2 = im2[0]

                except:
                    im2[1].verify('fix')

                    data2 = im2[1].data
                    checkdat2 = len(data2)

                    im2 = im2[1]


                while cutout_loop == 'y':

                    wcs = astropy.wcs.WCS(header_orig)

                    cutout = Cutout2D(data_orig, position=SNco, size=(cutoutsize_new,cutoutsize_new), wcs=wcs)

                    im_sci.data = cutout.data
                    im_sci.header.update(cutout.wcs.to_header())
                    im_sci.writeto('sci_trim.fits', overwrite=True)


                    cutout2 = Cutout2D(data2, position=SNco, size=(cutoutsize_new,cutoutsize_new), wcs=wcs)

                    im2.data = cutout2.data
                    im2.header.update(cutout2.wcs.to_header())
                    im2.writeto('tmpl_trim.fits', overwrite=True)

                    print('\nSubtracting template...')


                    try:
                        im_sub = run_subtraction('sci_trim.fits','tmpl_trim.fits','sci_psf.fits',
                        'tmpl_psf.fits',normalization='science',n_stamps=4,science_saturation=sci_sat_new, reference_saturation=tmpl_sat_new)
                                            
                    except:
                        if not quiet:
                            print('Subtraction failed - can vary parameters or proceed without subtraction')
                            
                            try_again = input('\nTry varying parameters? (y/n) [y] ')
                            if not try_again: try_again = 'y'

                            if try_again not in ('y','yes'):
                                print('\nUsing unsubtracted data')
                                template = ''
                                break
            
                            cutoutsize1 = input('Try larger cutout size? ['+str(cutoutsize_new)+']')
                            if not cutoutsize1: cutoutsize1 = cutoutsize_new
                            cutoutsize_new = int(cutoutsize1)
            
                            tmpl_sat1 = input('Try lower template saturation? ['+str(tmpl_sat_new)+']')
                            if not tmpl_sat1: tmpl_sat1 = tmpl_sat_new
                            tmpl_sat_new = int(tmpl_sat1)
            
                            sci_sat1 = input('Try lower science saturation? ['+str(sci_sat_new)+']')
                            if not sci_sat1: sci_sat1 = sci_sat_new
                            sci_sat_new = int(sci_sat1)
                            
                            continue
                        else:
                            print('\nUsing unsubtracted data')
                            template = ''
                            comment1 += 'subtraction failed'
                            do_sub = False
        
                            break

                    im_sub = np.real(im_sub[0])
                    
                    im_sci.data = im_sub
                    im_sci.writeto('sub.fits', overwrite=True)
                
                    data_sub = im_sub
                    
                    try:
                        bkg_new = photutils.background.Background2D(data_sub,box_size=bkgbox)
                    except:
                        bkg_new = photutils.background.Background2D(data_sub,box_size=int(cutoutsize_new/4),exclude_percentile=0)

                    
                    bkg_new_error = bkg_new.background_rms
                    
                    data_sub -= bkg_new.background

                    plt.figure(1)

                    ax1.clear()
                    
                    ax1.imshow(data_sub, origin='lower',cmap='gray',
                                vmin=visualization.ZScaleInterval().get_limits(data_sub)[0],
                                vmax=visualization.ZScaleInterval().get_limits(data_sub)[1])
                                
                    ax1.errorbar(co[:,0]-(SNco[0]-cutoutsize_new/2.),co[:,1]-(SNco[1]-cutoutsize_new/2.), fmt='s',mfc='none',markeredgecolor='C0',markersize=8,markeredgewidth=1.5)



                    ax1.set_title('Template-subtracted cutout')

                    ax1.set_xlim(0,len(data_sub))
                    ax1.set_ylim(0,len(data_sub))

                    ax1.get_yaxis().set_visible(False)
                    ax1.get_xaxis().set_visible(False)

                    

                    ax1.errorbar(cutoutsize_new/2.,cutoutsize_new/2.,fmt='o',markeredgecolor='r',mfc='none',
                                    markeredgewidth=3,markersize=20)
                                    
                    
                    plt.draw()

                    do_sub = True
                    
                    if not quiet:
                        happy = input('\nProceed with this subtraction? [y] ')
                        if not happy: happy = 'y'
                        
                        if happy not in ('y','yes'):
                        
                            try_again = input('\nTry varying parameters? (y/n) [y] ')
                            if not try_again: try_again = 'y'

                            if try_again not in ('y','yes'):
                                print('\nUsing unsubtracted data')
                                template = ''
                                do_sub = False
            
                                break

                            cutoutsize1 = input('Try larger cutout size? ['+str(cutoutsize_new)+']')
                            if not cutoutsize1: cutoutsize1 = cutoutsize_new
                            cutoutsize_new = int(cutoutsize1)
            
                            tmpl_sat1 = input('Try lower template saturation? ['+str(tmpl_sat_new)+']')
                            if not tmpl_sat1: tmpl_sat1 = tmpl_sat_new
                            tmpl_sat_new = int(tmpl_sat1)
            
                            sci_sat1 = input('Try lower science saturation? ['+str(sci_sat_new)+']')
                            if not sci_sat1: sci_sat1 = sci_sat_new
                            sci_sat_new = int(sci_sat1)
                            
                            continue
                        else:
                            do_sub = True
                            cutout_loop = 'n'

                    else:
                        do_sub = True
                        cutout_loop = 'n'


                if do_sub == True:
                    data = data_sub
                    bkg_error = bkg_new_error
                    SNco[0] = cutoutsize_new/2.
                    SNco[1] = cutoutsize_new/2.
                else:
                    plt.figure(1)

                    ax1.clear()
                    ax1.imshow(data, origin='lower',cmap='gray',
                                vmin=visualization.ZScaleInterval().get_limits(data)[0],
                                vmax=visualization.ZScaleInterval().get_limits(data)[1])

                    ax1.set_title(image+' ('+f+')')

                    ax1.set_xlim(0,len(data))
                    ax1.set_ylim(0,len(data))

                    ax1.get_yaxis().set_visible(False)
                    ax1.get_xaxis().set_visible(False)

                    ax1.errorbar(SNco[0],SNco[1],fmt='o',markeredgecolor='r',mfc='none',
                                    markeredgewidth=3,markersize=20)



        ########### SN photometry

            print('\nDoing photometry on science target...')

            if not forcepos:
                SNco[0],SNco[1] = photutils.centroids.centroid_sources(data,SNco[0],SNco[1],
                                             centroid_func=photutils.centroids.centroid_2dg)

            plt.figure(1)

            ax4 = plt.subplot2grid((2,5),(1,3))


            ax4.imshow(data, origin='lower',cmap='gray',
                        vmin=visualization.ZScaleInterval().get_limits(data)[0],
                        vmax=visualization.ZScaleInterval().get_limits(data)[1])

            ax4.set_xlim(SNco[0]-(aprad+skyrad),SNco[0]+(aprad+skyrad))
            ax4.set_ylim(SNco[1]-(aprad+skyrad),SNco[1]+(aprad+skyrad))

            ax4.get_yaxis().set_visible(False)
            ax4.get_xaxis().set_visible(False)

            ax4.set_title('Target')

            apcircle = Circle((SNco[0], SNco[1]), aprad_opt, facecolor='none',
                    edgecolor='r', linewidth=3, alpha=1)
            ax4.add_patch(apcircle)

            skycircle1 = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
                    edgecolor='r', linewidth=2, alpha=1)
            skycircle2 = Circle((SNco[0], SNco[1]), aprad+skyrad, facecolor='none',
                    edgecolor='r', linewidth=2, alpha=1)
            ax4.add_patch(skycircle1)
            ax4.add_patch(skycircle2)

            plt.draw()


            # apertures
    #        photap = photutils.CircularAperture(SNco, r=aprad_opt)
            photap = [photutils.CircularAperture(SNco, r=r) for r in [aprad_opt,aprad]]
            skyap = photutils.CircularAnnulus(SNco, r_in=aprad, r_out=aprad+skyrad)
            skymask = skyap.to_mask(method='center')

            # Get median sky in annulus around transient
            skydata = skymask.multiply(data)
            skydata_1d = skydata[skymask.data > 0]
            meansky, mediansky, sigsky = astropy.stats.sigma_clipped_stats(skydata_1d)
            bkg_local = np.array([mediansky])
            try:
                err_array = calc_total_error(data, bkg_error, gain)
            except:
                data_adu = data.copy()
                data = np.array([i.astype(float) for i in data_adu])
                bkg_error_adu = bkg_error.copy()
                bkg_error = np.array([i.astype(float) for i in bkg_error_adu])
                err_array = calc_total_error(data, bkg_error, gain)
            SNphotTab = photutils.aperture_photometry(data, photap, err_array)
            SNphotTab['local_sky'] = bkg_local
            SNphotTab['aperture_sum_sub'] = SNphotTab['aperture_sum_1'] - bkg_local * photap[1].area
            SNphotTab['aperture_opt_sum_sub'] = SNphotTab['aperture_sum_0'] - bkg_local * photap[0].area

            print('Aperture done')

            # PSF phot on transient
            SNcoordTable = astropy.table.Table()
            SNcoordTable['x_0'] = [SNco[0]]
            SNcoordTable['y_0'] = [SNco[1]]
            SNcoordTable['flux_0'] = SNphotTab['aperture_sum_sub']

    #        epsf.x_0.fixed = True
    #        epsf.y_0.fixed = True
            psfphot = photutils.psf.BasicPSFPhotometry(group_maker=grouper,
                            bkg_estimator=photutils.background.MMMBackground(),
                            psf_model=epsf, fitshape=fitrad,
                            finder=None, aperture_radius=min(stamprad,2*aprad_opt))

            SNpsfphotTab = psfphot.do_photometry(data, init_guesses=SNcoordTable)

            SNpsfsubIm = psfphot.get_residual_image()

            print('PSF done')

            ax5 = plt.subplot2grid((2,5),(1,4))

            ax5.imshow(SNpsfsubIm, origin='lower',cmap='gray',
                        vmin=visualization.ZScaleInterval().get_limits(data)[0],
                        vmax=visualization.ZScaleInterval().get_limits(data)[1])


            ax5.set_xlim(SNco[0]-(aprad+skyrad),SNco[0]+(aprad+skyrad))
            ax5.set_ylim(SNco[1]-(aprad+skyrad),SNco[1]+(aprad+skyrad))

            ax5.get_yaxis().set_visible(False)
            ax5.get_xaxis().set_visible(False)

            ax5.set_title('PSF subtracted')

            apcircle = Circle((SNco[0], SNco[1]), aprad_opt, facecolor='none',
                    edgecolor='r', linewidth=3, alpha=1)
            ax5.add_patch(apcircle)

            skycircle1 = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
                    edgecolor='r', linewidth=2, alpha=1)
            skycircle2 = Circle((SNco[0], SNco[1]), aprad+skyrad, facecolor='none',
                    edgecolor='r', linewidth=2, alpha=1)
            ax5.add_patch(skycircle1)
            ax5.add_patch(skycircle2)

            plt.draw()

            plt.tight_layout(pad=0.5)

            plt.subplots_adjust(hspace=0.1,wspace=0.2)
            
            
            # Convert flux to instrumental magnitudes

            print('Converting flux to magnitudes...')

            SNap = -2.5*np.log10(SNphotTab['aperture_sum_sub'])
            # aperture mag error assuming Poisson noise
            errSNap = abs(SNphotTab['aperture_sum_err_1'] / SNphotTab['aperture_sum_sub'])

            SNap_opt = -2.5*np.log10(SNphotTab['aperture_opt_sum_sub'])
            SNap_opt_corr = -2.5*np.log10(SNphotTab['aperture_opt_sum_sub']) + ap_corr
            # aperture mag error assuming Poisson noise
            errSNap_opt = abs(SNphotTab['aperture_sum_err_0'] / SNphotTab['aperture_opt_sum_sub'])
            errSNap_opt_corr = np.sqrt((abs(SNphotTab['aperture_sum_err_0'] / SNphotTab['aperture_opt_sum_sub']))**2 + (0.1*ap_corr)**2)

            ulim = -2.5*np.log10(3*np.sqrt(sigsky) * photap[0].area)

            try:
                SNpsf = -2.5*np.log10(SNpsfphotTab['flux_fit'])
                errSNpsf = abs(SNpsfphotTab['flux_unc']/SNpsfphotTab['flux_fit'])
            except:
                SNpsf = np.nan
                errSNpsf = np.nan
                print('PSF fit failed, aperture mag only!')

            print('\n')

            if f in seqMags:

                calMagPsf = SNpsf + ZP_psf

                errMagPsf = np.sqrt(errSNpsf**2 + errZP_psf**2)


                calMagAp = SNap + ZP_ap

                errMagAp = np.sqrt(errSNap**2 + errZP_ap**2)
                
                
                calMagAp_opt = SNap_opt + ZP_opt

                errMagAp_opt = np.sqrt(errSNap_opt**2 + errZP_opt**2)

                
                calMagLim = ulim + ZP_opt

            else:
                calMagPsf = SNpsf

                errMagPsf = errSNpsf


                calMagAp = SNap

                errMagAp = errSNap
                
                
                calMagAp_opt = SNap_opt

                errMagAp_opt = errSNap_opt

                
                calMagLim = ulim

                comment1 += ' instrumental mag only'


            print('> PSF mag = '+'%.2f +/- %.2f' %(calMagPsf,errMagPsf))
            print('> Aperture mag (optimised aperture) = '+'%.2f +/- %.2f' %(calMagAp_opt,errMagAp_opt))
            print('> Aperture mag (big aperture) = '+'%.2f +/- %.2f' %(calMagAp,errMagAp))
            print('> Limiting mag (3 sigma, optimum ap) = '+'%.2f' %(calMagLim))

            comment = ''
            if not quiet:
                comment = input('\n> Add comment to output file: ')

            if comment1:
                comment += (' // '+comment1)

            outFile.write('\n'+image+'\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\t%s' %(target_name,f,mjd,calMagPsf,errMagPsf,calMagAp_opt,errMagAp_opt,calMagAp,errMagAp,calMagLim,faintest_object,ZP,errZP,template,comment))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print('Error encountered (skipping): image '+image)
            print(e)
            print('Line number:')
            print(exc_tb.tb_lineno)
            outFile.write('\n'+image+'\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tFAILED')
            if not quiet:
                next = input('\n> Press enter to continue to next image')

outFile.close()



print('\n##########################################\nFinished!\nResults saved to \n'+results_filename+ '\n##########################################')

if not keep:
    if os.path.exists('tmpl_aligned.fits'):
        os.remove('tmpl_aligned.fits')
    if os.path.exists('tmpl_trim.fits'):
        os.remove('tmpl_trim.fits')
    if os.path.exists('tmpl_psf.fits'):
        os.remove('tmpl_psf.fits')
    if os.path.exists('sci_trim.fits'):
        os.remove('sci_trim.fits')
    if os.path.exists('sci_psf.fits'):
        os.remove('sci_psf.fits')
    if os.path.exists('sub.fits'):
        os.remove('sub.fits')

