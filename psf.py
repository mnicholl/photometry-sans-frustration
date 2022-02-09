#!/usr/bin/env python

version = '1.0'

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
from skimage.registration import phase_cross_correlation
from scipy.ndimage import interpolation as interp
import time
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import astropy.units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
import astroalign as aa
#from reproject import reproject_interp
from PyZOGY.subtract import run_subtraction
from photutils.utils import calc_total_error
import warnings


warnings.filterwarnings("ignore")



# Optional flags:

parser = argparse.ArgumentParser()

parser.add_argument('--ims','-i', dest='file_to_reduce', default='', nargs='+',
                    help='List of files to reduce. Accepts wildcards or '
                    'space-delimited list.')

parser.add_argument('--magmin', dest='magmin', default=20.5, type=float,
                    help='Faintest sequence stars to use ')

parser.add_argument('--magmax', dest='magmax', default=16.5, type=float,
                    help='Brightest sequence stars to use ')

parser.add_argument('--shifts', dest='shifts', default=False, action='store_true',
                    help='Apply manual shifts if WCS is a bit off ')

parser.add_argument('--ap', dest='aprad', default=15, type=int,
                    help='Radius for aperture/PSF phot.')

parser.add_argument('--sky', dest='skyrad', default=5, type=int,
                    help='Width of annulus for sky background.')

parser.add_argument('--box', dest='bkgbox', default=500, type=int,
                    help='Size of stamps for background fit.')

parser.add_argument('--psfthresh', dest='psfthresh', default=20., type=float,
                    help='SNR threshold for inclusion in PSF model.')

parser.add_argument('--zpsig', dest='sigClip', default=1, type=int,
                    help='Sigma clipping for rejecting sequence stars.')

parser.add_argument('--samp', dest='samp', default=1, type=int,
                    help='Oversampling factor for PSF build.')

parser.add_argument('--quiet', dest='quiet', default=False, action='store_true',
                    help='Run with no user prompts')

parser.add_argument('--stack', dest='stack', default=False, action='store_true',
                    help='Stack images that are in the same filter')

parser.add_argument('--sub', dest='sub', default=False, action='store_true',
                    help='Subtract template images')

parser.add_argument('--keep', dest='keep', default=False, action='store_true',
                    help='Keep intermediate products')




args = parser.parse_args()

magmin = args.magmin
magmax = args.magmax
shifts = args.shifts
aprad = args.aprad
skyrad = args.skyrad
bkgbox = args.bkgbox
psfthresh = args.psfthresh
sigClip = args.sigClip
samp = args.samp
quiet = args.quiet
stack = args.stack
sub = args.sub
keep = args.keep

ims = [i for i in args.file_to_reduce]

# If no images provided, run on all images in directory
if len(ims) == 0:
    ims = glob.glob('*.fits')


##################################################



##### FUNCTIONS TO QUERY PANSTARRS #######

def PS1catalog(ra,dec,magmin=25,magmax=8):

    queryurl = 'https://archive.stsci.edu/panstarrs/search.php?'
    queryurl += 'RA='+str(ra)
    queryurl += '&DEC='+str(dec)
    queryurl += '&SR=0.083&selectedColumnsCsv=ndetections,raMean,decMean,'
    queryurl += 'gMeanPSFMag,rMeanPSFMag,iMeanPSFMag,zMeanPSFMag,yMeanPSFMag,iMeanKronMag'
    queryurl += '&ordercolumn1=ndetections&descending1=on&max_records=200'

    print('\nQuerying PS1 for reference stars via MAST...\n')

    query = requests.get(queryurl)

    results = query.text

    entries = results.split('DATA')[2][11:][:-19].split('</TD>\n</TR>\n<TR>\n<TD>')

    data = []

    for i in entries:
        data.append(np.array(i.split('</TD><TD>')).T)

    if len(data) > 1:

        data = np.array(data).astype(float)

        # Get rid of n_det column
        data = data[:,1:][data[:,0]>3]

#        # Get rid of non-detections:
#        data = data[data[:,2]>-999]
#        data = data[data[:,3]>-999]
#        data = data[data[:,4]>-999]
#        data = data[data[:,5]>-999]
#        data = data[data[:,6]>-999]
#
#        # Get rid of very faint stars
#        data = data[data[:,2]<magmin]
#        data = data[data[:,3]<magmin]
#        data = data[data[:,4]<magmin]
#        data = data[data[:,5]<magmin]
#        data = data[data[:,6]<magmin]
#
#        # Get rid of stars likely to saturate
#        data = data[data[:,2]>magmax]
#        data = data[data[:,3]>magmax]
#        data = data[data[:,4]>magmax]
#        data = data[data[:,5]>magmax]
#        data = data[data[:,6]>magmax]


        # Star-galaxy separation
        data = data[:,:-1][data[:,4]-data[:,-1]<0.05]

        np.savetxt('PS1_seq.txt',data,fmt='%.8f\t%.8f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f', header='ra\tdec\tg\tr\ti\tz\ty',comments='')

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
        print('\SDSS template search failed!\n')
        
        
def SDSScatalog(ra,dec,magmin=25,magmax=8):
 
    print('\nQuerying SDSS for reference stars via Astroquery...\n')

    pos = coords.SkyCoord(str(ra)+' '+str(dec), unit='deg', frame='icrs')
    
    data = SDSS.query_region(pos,radius='0.1d',fields=('ra','dec','u','g','r','i','z','type'))
    
    data = data[data['type'] == 6]

#    # Get rid of non-detections:
#    data = data[data['u']>-9999]
#    data = data[data['g']>-9999]
#    data = data[data['r']>-9999]
#    data = data[data['i']>-9999]
#    data = data[data['z']>-9999]
#
#    # Get rid of very faint stars
#    data = data[data['u']<magmin]
#    data = data[data['g']<magmin]
#    data = data[data['r']<magmin]
#    data = data[data['i']<magmin]
#    data = data[data['z']<magmin]
#
#    # Get rid of stars likely to saturate
#    data = data[data['u']>magmax]
#    data = data[data['g']>magmax]
#    data = data[data['r']>magmax]
#    data = data[data['i']>magmax]
#    data = data[data['z']>magmax]
    
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

filtAll = 'ugrizUBVRIJHK'



print('#################################################\n#                                               #\n#  Welcome to PSF: Photometry Sans Frustration  #\n#                    (V'+version+')                     #\n#        Written by Matt Nicholl (2015)         #\n#                                               #\n#################################################')


# A place to save output files
outdir = 'PSF_output_'+str(int(time.time()))

if not os.path.exists(outdir): os.makedirs(outdir)

# A file to write final magnitudes
outFile = open(os.path.join(outdir,'PSF_phot_'+str(len(glob.glob(os.path.join(outdir,'PSF_phot_*'))))+'.txt'),'w')
outFile.write('#image\tfilter\tmjd\tPSFmag\terr\tAPmag\terr\tZP\terr\ttemplate\tcomments')



# Prepare to plot images and PSFs

plt.figure(1,(14,7))
plt.ion()
plt.show()



################################
# Part one: get coordinates
################################

# Search for SN coordinates file (1 line: RA Dec) in this directory and parents

suggSn = glob.glob('*coords.txt')
if len(suggSn)==0:
    suggSn = glob.glob('../*coords.txt')
if len(suggSn)==0:
    suggSn = glob.glob('../../*coords.txt')

# Needs SN coords to run!
if len(suggSn)>0:
    snFile = suggSn[0]
else:
    sys.exit('Error: no SN coordinates (*_coords.txt) found')

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

    try:
        mjd = fits.getval(image,'MJD')
    except:
        try:
            mjd = fits.getval(image,'MJD-OBS')
        except:
            try:
                jd = fits.getval(image,'JD')
                mjd = jd - 2400000
            except:
                mjd = 99999

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
                    PS1cutouts(RAdec[0],RAdec[1],filtername)
                    template = 'template_'+filtername+'.fits'
                except:
                    print('Error: could not match template from PS1')
            elif filtername == 'u':
                try:
                    SDSScutouts(RAdec[0],RAdec[1],filtername)
                    template = 'template_'+filtername+'.fits'
                except:
                    print('Error: could not match template from SDSS')


    filtertab.append([image, filtername, mjd, template])

    if not filtername in usedfilters:
        usedfilters.append(filtername)

filtertab = np.array(filtertab)



#################################
# Part three: do some photometry
#################################

for f in usedfilters:

    print('\n\n#########\n'+f+'-band\n#########')

    ims1 = filtertab[:,0][filtertab[:,1]==f]
    
    templates1 = filtertab[:,-1][filtertab[:,1]==f]
        
    if sub==True:
        if len(np.unique(templates1)) > 1:
            print('Warning: different templates for same filter!')
            print('Using '+templates1[0])
        
        template = templates1[0]

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




    zero_shift_image = ''
    
    dostack = 'n'

    if stack==True:
        dostack = 'y'
        if len(ims1) < 2:
            print('\nOnly 1 image in filter, skipping stacking!')
            dostack = 'n'
        else:
            if os.path.exists('stack_'+f+'.fits'):
                if not quiet:
                    dostack = input('\nStack already found in filter, overwrite? [n]')
                    if not dostack: dostack = 'n'
                else:
                    dostack = 'n'
                ims1 = ['stack_'+f+'.fits']
                
    if dostack in ('y','yes'):
        print('\nAligning and stacking images...')
        
        mjdstack = np.mean(filtertab[:,2][filtertab[:,1]==f].astype(float))

        zero_shift_image = ims1[0]
        
        imshifts = {} # dictionary to hold the x and y shift pairs for each image
        for im1 in ims1:
            ## phase_cross_correlation is a function that calculates shifts by comparing 2-D arrays
            imshift, imshifterr, diffphase = phase_cross_correlation(
                fits.getdata(zero_shift_image),
                fits.getdata(im1))
            imshifts[im1] = imshift

        ## new dictionary for shifted image data:
        shifted_data = {}
        for im1 in imshifts:
            shifted_data[im1] = interp.shift(
                fits.getdata(im1),
                imshifts[im1])
            shifted_data[im1][shifted_data[im1] == 0] = 'nan'

        shifted_data_cube = np.stack([shifted_data[im1] for im1 in ims1])
        stacked_data = np.nanmedian(shifted_data_cube, axis=0)
        
        stackheader = fits.getheader(zero_shift_image)
        
        stackheader['MJD'] = mjdstack
        stackheader['MJD-OBS'] = mjdstack

        fits.writeto('stack_'+f+'.fits',
                stacked_data,header=stackheader,
                overwrite=True)

        print('Done')

        ### NEED TO MAKE COMPATIBLE WITH NAMES OF IMAGE IN SUBTRACTION PART

        ims2 = ['stack_'+f+'.fits']
    else:
        ims2 = ims1.copy()

    for image in ims2:
    
        plt.clf()


        print('\n> Image: '+image)

        if image == 'stack_'+f+'.fits':
            mjd = fits.getval(filename='stack_'+f+'.fits',keyword='MJD')
        else:
            mjd = filtertab[:,2][filtertab[:,0]==image][0]

        comment1 = ''


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


        # Set up sequence stars, initial steps
        
        mag_range = (seqMags[f]>magmax)&(seqMags[f]<magmin)
        
        co = astropy.wcs.WCS(header=header).all_world2pix(seqDat[:,0],seqDat[:,1],1)
        co = np.array(list(zip(co[0],co[1])))
        
        co = co[mag_range]

        # Remove any stars falling outside the image or too close to edge
        inframe = (co[:,0]>2*aprad)&(co[:,0]<len(data[0])-2*aprad)&(co[:,1]>2*aprad)&(co[:,1]<len(data)-2*aprad)
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

        found = (abs(co[:,0]-orig_co[:,0])<abs(del_x)*10)&(abs(co[:,1]-orig_co[:,1])<abs(del_y)*10)

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

            # extract stars from image
            psfstars = photutils.psf.extract_stars(nddata, psfinput[photTab['aperture_sum']>psfthresh*photTab['aperture_sum_err']], size=2*aprad+5)
            while(len(psfstars))<3:
                print('Warning: too few PSF stars with threshold '+str(psfthresh)+' sigma, trying lower sigma)')
                psfthresh -= 1
                psfstars = photutils.psf.extract_stars(nddata, psfinput[photTab['aperture_sum']>psfthresh*photTab['aperture_sum_err']], size=2*aprad+5)

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

            ax1.legend(frameon=True,fontsize=16)


            # build PSF
            epsf_builder = photutils.EPSFBuilder(maxiters=10,recentering_maxiters=5,
                            oversampling=samp,smoothing_kernel='quadratic',shape=2*aprad-1)
            epsf, fitted_stars = epsf_builder(psfstars)

            psf = epsf.data

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


            if not quiet:
                happy = input('\nProceed with this PSF? [y] ')
                if not happy: happy = 'y'
                if happy not in ('y','yes'):
                    aprad1 = input('Try new aperture radius: [' +str(aprad)+']')
                    if not aprad1: aprad1 = aprad
                    aprad = int(aprad1)
                    
                    psfthresh1 = input('Try new inclusion threshold: [' +str(psfthresh)+' sigma]')
                    if not psfthresh1: psfthresh1 = psfthresh
                    psfthresh = int(psfthresh1)
                    
                    samp1 = input('Try new PSF oversampling: [' +str(samp)+']')
                    if not samp1: samp1 = samp
                    samp = int(samp1)
            else:
                happy = 'y'
                
                
        scipsf = fits.PrimaryHDU(psf)
        scipsf.writeto('sci_psf.fits',overwrite=True)

        print('\nStarting PSF photometry...')

        psfcoordTable = astropy.table.Table()

        psfcoordTable['x_0'] = co[:,0]
        psfcoordTable['y_0'] = co[:,1]
        psfcoordTable['flux_0'] = photTab['aperture_sum']

        grouper = photutils.psf.DAOGroup(crit_separation=aprad)

        # need an odd number of pixels to fit PSF
        fitrad = 2*aprad + 1

        psfphot = photutils.psf.BasicPSFPhotometry(group_maker=grouper,
                        bkg_estimator=photutils.background.MMMBackground(),
                        psf_model=epsf, fitshape=fitrad,
                        finder=None, aperture_radius=aprad)

        psfphotTab = psfphot.do_photometry(data, init_guesses=psfcoordTable)

        psfsubIm = psfphot.get_residual_image()

        ax1.imshow(psfsubIm, origin='lower',cmap='gray',
                    vmin=visualization.ZScaleInterval().get_limits(data)[0],
                    vmax=visualization.ZScaleInterval().get_limits(data)[1])

        goodStars = (photTab['aperture_sum']>5*photTab['aperture_sum_err'])&(psfphotTab['flux_fit']/psfphotTab['flux_0']>0.5)&(psfphotTab['flux_fit']/psfphotTab['flux_0']<1.5)

        ax1.errorbar(co[:,0][~goodStars],co[:,1][~goodStars],fmt='x',mfc='none',
                    markeredgewidth=2, color='C1',
                    markersize=8,label='Too faint or PSF/ap differ')
                    
        ax1.legend(frameon=True,fontsize=16)

#        ax1.text(len(data[:,0])*0.8,len(data[:,1])*0.9,'Rejected',color='C1')

        print('Done')

    ########## Zero point from seq stars

        print('\nComputing image zeropoint...')

        if f in seqMags:
        
            magmax2 = magmax
            magmin2 = magmin
            
            happy = 'n'
            while happy not in ('y','yes'):
            
                mag_range_2 = (seqMags[f][mag_range][inframe][goodpix][found][goodStars]>magmax2) & (seqMags[f][mag_range][inframe][goodpix][found][goodStars]<magmin2)
            
                flux = np.array(psfphotTab['flux_fit'])[goodStars]
                seqIm = -2.5*np.log10(flux)

                zpList1 = seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2] - seqIm[mag_range_2]

                axZP = plt.subplot2grid((2,5),(1,2))

                axZP.scatter(seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2], zpList1,color='r')

                zp1 = np.nanmean(zpList1)
                errzp1 = np.nanstd(zpList1)

                print('\nInitial zeropoint =  %.2f +/- %.2f\n' %(zp1,errzp1))
                print('Checking for bad stars...')

                checkMags = np.abs(seqIm[mag_range_2]+zp1 - seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2]) < errzp1*sigClip

#                if False in checkMags:
#                    print('Rejecting stars from ZP: ')
#                    print(psfphotTab['id'][goodStars][mag_range_2][~checkMags])
#                    print('Bad ZPs:')
#                    print( seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][~checkMags] - seqIm[mag_range_2][~checkMags])

                ax1.errorbar(co[:,0][goodStars][mag_range_2][~checkMags],
                            co[:,1][goodStars][mag_range_2][~checkMags],fmt='x',mfc='none',
                            markeredgewidth=2, color='C3',
                            markersize=8,label='Sigma clipped from ZP')

                ax1.legend(frameon=True,fontsize=16)

                zpList = seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][checkMags] - seqIm[mag_range_2][checkMags]

                ZP = np.mean(zpList)
                errZP = np.std(zpList)/np.sqrt(len(zpList))
                
                axZP.scatter(seqMags[f][mag_range][inframe][goodpix][found][goodStars][mag_range_2][checkMags], zpList,color='k')

                axZP.axhline(ZP,linestyle='-',color='C0')
                axZP.axhline(ZP-errZP,linestyle='--',color='C0')
                axZP.axhline(ZP+errZP,linestyle='--',color='C0')

                axZP.set_xlabel('Magnitude')
                axZP.set_title('Zero point')
                
                axZP.set_ylim(max(max(zpList1)+0.5,ZP+1),min(min(zpList1)-0.5,ZP-1))
                axZP.set_xlim(magmax2,magmin2)
                
                plt.draw()

                print('\nFinal Zeropoint = %.2f +/- %.2f' %(ZP,errZP))

                
                if not quiet:
                    happy = input('\nProceed with this zeropoint? [y] ')
                    if not happy: happy = 'y'
                    if happy not in ('y','yes'):
                        magmax1 = input('Use new maximum mag: [' +str(magmax2)+']')
                        if not magmax1: magmax1 = magmax2
                        magmax2 = float(magmax1)
                                                
                        magmin1 = input('Use new minimum mag: [' +str(magmin2)+']')
                        if not magmin1: magmin1 = magmin2
                        magmin2 = float(magmin1)
                else:
                    happy = 'y'


        else:
            ZP = np.nan
            errZP = np.nan

            print('\nCould not determine ZP (no sequence star mags in filter?) : instrumental mag only!!!')

        
    ########### Template subtraction
    
        if sub == True:
 
            if not quiet:
                conti = input('\n > Proceed to template image... (return)')

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


            ### Using Reproject

#            tmp_resampled, footprint = reproject_interp(tmp, header)
#
#            tmp_resampled[np.isnan(tmp_resampled)] = np.nanmedian(tmp_resampled)
#
#            hdu2 = fits.PrimaryHDU(tmp_resampled)
#
#            hdu2.writeto('tmpl_aligned.fits',overwrite=True)



            ### Using Astroalign

            im_fixed = np.array(data, dtype="<f4")
            tmp_fixed = np.array(data2, dtype="<f4")
    
            registered, footprint = aa.register(tmp_fixed, im_fixed)
    
            tmp_masked = np.ma.masked_array(registered, footprint, fill_value=np.nanmedian(tmp_fixed)).filled()
    
            tmp_masked[np.isnan(tmp_masked)] = np.nanmedian(tmp_fixed)
    
            hdu2 = fits.PrimaryHDU(tmp_masked)
    
            hdu2.writeto('tmpl_aligned.fits',overwrite=True)
            


            fig2 = plt.figure(2,(14,7))
            plt.clf()
            plt.ion()
            plt.show()

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

            err_array2 = calc_total_error(data2, bkg_error2, gain2)


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

            aprad2 = aprad
            psfthresh2 = psfthresh
            samp2 = samp

            # Create model from sequence stars
            happy = 'n'
            while happy not in ('y','yes'):

                psfstars2 = photutils.psf.extract_stars(nddata2, psfinput2[photTab2['aperture_sum']>psfthresh2*photTab2['aperture_sum_err']], size=2*aprad2+5)
                while(len(psfstars2))<3:
                    print('Warning: too few PSF stars with threshold '+str(psfthresh2)+' sigma, trying lower sigma)')
                    psfthresh2 -= 1
                    psfstars2 = photutils.psf.extract_stars(nddata2, psfinput2[photTab2['aperture_sum']>psfthresh2*photTab2['aperture_sum_err']], size=2*aprad2+5)


                ax1t.errorbar(psfstars2.center_flat[:,0],psfstars2.center_flat[:,1],fmt='*',mfc='none', markeredgecolor='lime',markeredgewidth=2, markersize=20,label='Used in PSF fit')
                
                
                # build PSF
                epsf_builder = photutils.EPSFBuilder(maxiters=10,recentering_maxiters=5,
                                oversampling=samp2,smoothing_kernel='quadratic',shape=2*aprad2-1)
                epsf2, fitted_stars2 = epsf_builder(psfstars2)

                psf2 = epsf2.data

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


                if not quiet:
                    happy = input('\nProceed with this template PSF? [y] ')
                    if not happy: happy = 'y'
                    if happy != 'y':
                        aprad1 = input('Try new aperture radius: [' +str(aprad2)+']')
                        if not aprad1: aprad1 = aprad2
                        aprad2 = int(aprad1)
                        
                        psfthresh1 = input('Try new inclusion threshold: [' +str(psfthresh2)+' sigma]')
                        if not psfthresh1: psfthresh1 = psfthresh2
                        psfthresh2 = int(psfthresh1)
                        
                        samp1 = input('Try new PSF oversampling: [' +str(samp2)+']')
                        if not samp1: samp1 = samp2
                        samp2 = int(samp1)
                else:
                    happy = 'y'


            tmppsf = fits.PrimaryHDU(psf2)
            tmppsf.writeto('tmpl_psf.fits',overwrite=True)

            
            plt.close(fig2)
            
            
            # Make cutouts for subtraction
            
            im_sci = fits.open(image)

            try:
                im_sci[0].verify('fix')

                data_orig = im_sci[0].data
                header_orig = im_sci[0].header
                checkdat = len(data_orig)
                
                im_sci = im_sci[0]

            except:
                im_sci[1].verify('fix')

                data_orig = im_sci[1].data
                header_orig = im_sci[1].header
                checkdat = len(data_orig)
                
                im_sci = im_sci[1]


            wcs = astropy.wcs.WCS(header_orig)

            cutoutsize = 1000

            cutout = Cutout2D(data_orig, position=SNco, size=(cutoutsize,cutoutsize), wcs=wcs)

            im_sci.data = cutout.data
            im_sci.header.update(cutout.wcs.to_header())
            im_sci.writeto('sci_trim.fits', overwrite=True)


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


            cutout2 = Cutout2D(data2, position=SNco, size=(cutoutsize,cutoutsize), wcs=wcs)

            im2.data = cutout2.data
            im2.header.update(cutout2.wcs.to_header())
            im2.writeto('tmpl_trim.fits', overwrite=True)

            print('\nSubtracting template...')

            im_sub = run_subtraction('sci_trim.fits','tmpl_trim.fits','sci_psf.fits',
                'tmpl_psf.fits',normalization="science",n_stamps=4)
            
            im_sub = np.real(im_sub[0])
            
            im_sci.data = im_sub
            im_sci.writeto('sub.fits', overwrite=True)
        
            data = im_sub
            
            bkg_new = photutils.background.Background2D(data,box_size=bkgbox)
            
            bkg_error = bkg_new.background_rms
            
            data -= bkg_new.background
            
            ax1.clear()
            
            ax1.imshow(data, origin='lower',cmap='gray',
                        vmin=visualization.ZScaleInterval().get_limits(data)[0],
                        vmax=visualization.ZScaleInterval().get_limits(data)[1])
                        
            ax1.errorbar(co[:,0]-(SNco[0]-cutoutsize/2.),co[:,1]-(SNco[1]-cutoutsize/2.), fmt='s',mfc='none',markeredgecolor='C0',markersize=8,markeredgewidth=1.5)



            ax1.set_title('Template-subtracted cutout')

            ax1.set_xlim(0,len(data))
            ax1.set_ylim(0,len(data))

            ax1.get_yaxis().set_visible(False)
            ax1.get_xaxis().set_visible(False)

            
            SNco[0] = cutoutsize/2.
            SNco[1] = cutoutsize/2.

            ax1.errorbar(SNco[0],SNco[1],fmt='o',markeredgecolor='r',mfc='none',
                            markeredgewidth=3,markersize=20)

    ########### SN photometry

        print('\nDoing photometry on science target...')

        SNco[0] += del_x
        SNco[1] += del_y

        # SNco[0],SNco[1] = photutils.centroids.centroid_sources(data,SNco[0],SNco[1],
        #                             centroid_func=photutils.centroids.centroid_2dg)
#        SNco = [np.array([SNco[0]]),np.array([SNco[1]])]

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

        apcircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
                edgecolor='r', linewidth=3, alpha=1)
        ax4.add_patch(apcircle)

        skycircle = Circle((SNco[0], SNco[1]), aprad+skyrad, facecolor='none',
                edgecolor='r', linewidth=3, alpha=1)
        ax4.add_patch(skycircle)

        plt.draw()


        # apertures
        photap = photutils.CircularAperture(SNco, r=aprad)
        skyap = photutils.CircularAnnulus(SNco, r_in=aprad, r_out=aprad+skyrad)
        skymask = skyap.to_mask(method='center')

        # Get median sky in annulus around transient
        skydata = skymask.multiply(data)
        skydata_1d = skydata[skymask.data > 0]
        meansky, mediansky, sigsky = astropy.stats.sigma_clipped_stats(skydata_1d)
        bkg_local = np.array([mediansky])
        err_array = calc_total_error(data, bkg_error, gain)
        SNphotTab = photutils.aperture_photometry(data, photap, err_array)
        SNphotTab['local_sky'] = bkg_local
        SNphotTab['aperture_sum_sub'] = SNphotTab['aperture_sum'] - bkg_local * photap.area

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
                        finder=None, aperture_radius=aprad)

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

        apcircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
                edgecolor='r', linewidth=3, alpha=1)
        ax5.add_patch(apcircle)

        skycircle = Circle((SNco[0], SNco[1]), aprad+skyrad, facecolor='none',
                edgecolor='r', linewidth=3, alpha=1)
        ax5.add_patch(skycircle)

        plt.draw()

        plt.tight_layout(pad=0.5)

        plt.subplots_adjust(hspace=0.1,wspace=0.2)
        
        
        # Convert flux to instrumental magnitudes

        print('Converting flux to magnitudes...')

        SNap = -2.5*np.log10(SNphotTab['aperture_sum_sub'])
        # aperture mag error assuming Poisson noise
        errSNap = 0.92*abs(SNphotTab['aperture_sum_err'] / SNphotTab['aperture_sum_sub'])

        try:
            SNpsf = -2.5*np.log10(SNpsfphotTab['flux_fit'])
            errSNpsf = 0.92*abs(SNpsf * SNpsfphotTab['flux_unc']/SNpsfphotTab['flux_fit'])
        except:
            SNpsf = np.nan
            errSNpsf = np.nan
            print('PSF fit failed, aperture mag only!')

        print('\n')

        if f in seqMags:

            calMagPsf = SNpsf + ZP

            errMagPsf = np.sqrt(errSNpsf**2 + errZP**2)


            calMagAp = SNap + ZP

            errMagAp = np.sqrt(errSNap**2 + errZP**2)

        else:
            calMagPsf = SNpsf

            errMagPsf = errSNpsf


            calMagAp = SNap

            errMagAp = errSNap

            comment1 = 'instrumental mag only'


        print('> PSF mag = '+'%.2f +/- %.2f' %(calMagPsf,errMagPsf))
        print('> Aperture mag = '+'%.2f +/- %.2f' %(calMagAp,errMagAp))

        comment = ''
        if not quiet:
            comment = input('\n> Add comment to output file: ')

        if comment1:
            comment += (' // '+comment1)

        outFile.write('\n'+image+'\t'+f+'\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\t%s'
                        %(mjd,calMagPsf,errMagPsf,calMagAp,errMagAp,ZP,errZP,template,comment))
#        outFile.write(comment)


outFile.close()


print('\n##########################################\nFinished!\nCalibrated PSF phot saved to ./PSF_phot.txt\nAperture photometry saved to ./ap_phot.txt\nCheck PSF_output/ for additional info\n##########################################')

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

