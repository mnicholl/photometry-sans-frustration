#!/usr/bin/env python

version = '0.2'

'''
    PSF: PHOTOMETRY SANS FRUSTRATION

    Written by Matt Nicholl, 2015-2019

    Requirements:

    Needs astropy, numpy, and matplotlib. Also requests if querying PanSTARRS.

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

    If sequence star file does not exist, code will create one from PS1 archive.
    But note that PS1 only contains grizy magnitudes!

    Given this list of field star magnitudes and coordinates, psf.py will
    compute the zeropoint of the image, construct a point spread function
    from these stars, fit this to the target of interest, show the resulting
    subtraction, and return the apparent magnitude from both PSF and aperture
    photometry

    Run with python psf.py <flags> (see help message with psf.py --help)

    Outputs a text file PSF_phot_X.txt (where X is an integer that increases
    every time code is run, to avoid overwriting previous results)
    Format of text file is:
        image  filter  mjd  PSFmag  err  APmag  err  comments
        - Row exists for each input image used in run
        - PSFmag is from PSF fitting, APmag is from simple aperture photometry
        using aperture size specified with --ap (default 10 pixel radius)
        - ZP is measured from sequence stars
        - comment allows user to specify if e.g. PSF fit looked unreliable

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


# Optional flags:

parser = argparse.ArgumentParser()

parser.add_argument('--ims','-i', dest='file_to_reduce', default='', nargs='+',
                    help='List of files to reduce. Accepts wildcards or '
                    'space-delimited list.')

parser.add_argument('--magmin', dest='magmin', default=21.5, type=float,
                    help='Faintest sequence stars to return from PS1 query ')

parser.add_argument('--magmax', dest='magmax', default=16.5, type=float,
                    help='Brightest sequence stars to return from PS1 query ')

parser.add_argument('--shifts', dest='shifts', default=False, action='store_true',
                    help='Apply manual shifts if WCS is a bit off ')

parser.add_argument('--ap', dest='aprad', default=15, type=int,
                    help='Radius for aperture/PSF phot.')

parser.add_argument('--sky', dest='skyrad', default=5, type=int,
                    help='Width of annulus for sky background.')

parser.add_argument('--box', dest='bkgbox', default=500, type=int,
                    help='Size of stamps for background fit.')

parser.add_argument('--zpsig', dest='sigClip', default=3, type=int,
                    help='Sigma clipping for rejecting sequence stars.')

parser.add_argument('--quiet', dest='quiet', default=False, action='store_true',
                    help='Run with no user prompts')



args = parser.parse_args()

magmin = args.magmin
magmax = args.magmax
shifts = args.shifts
aprad = args.aprad
skyrad = args.skyrad
bkgbox = args.bkgbox
sigClip = args.sigClip

ims = [i for i in args.file_to_reduce]

# If no images provided, run on all images in directory
if len(ims) == 0:
    ims = glob.glob('*.fits')


##################################################



##### FUNCTIONS TO QUERY PANSTARRS #######

def PS1catalog(ra,dec,magmin,magmax):

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

        # Get rid of non-detections:
        data = data[data[:,2]>-999]
        data = data[data[:,3]>-999]
        data = data[data[:,4]>-999]
        data = data[data[:,5]>-999]
        data = data[data[:,6]>-999]

        # Get rid of very faint stars
        data = data[data[:,2]<magmin]
        data = data[data[:,3]<magmin]
        data = data[data[:,4]<magmin]
        data = data[data[:,5]<magmin]
        data = data[data[:,6]<magmin]

        # Get rid of stars likely to saturate
        data = data[data[:,2]>magmax]
        data = data[data[:,3]>magmax]
        data = data[data[:,4]>magmax]
        data = data[data[:,5]>magmax]
        data = data[data[:,6]>magmax]


        # Star-galaxy separation
        data = data[:,:-1][data[:,4]-data[:,-1]<0.05]

        np.savetxt('PS1_seq.txt',data,fmt='%.8f\t%.8f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f',header='Ra\tDec\tg\tr\ti\tz\ty\n',comments='')

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
        image_name = ps1_im.text.split()[16]

        print('Image found: ' + image_name + '\n')

        cutout_url = 'http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?&filetypes=stack&size=2500'

        cutout_url += '&ra='+str(ra)
        cutout_url += '&dec='+str(dec)
        cutout_url += '&filters='+filt
        cutout_url += '&format=fits'
        cutout_url += '&red='+image_name

        dest_file = filt + '_template.fits'

        cmd = 'wget -O %s "%s"' % (dest_file, cutout_url)

        os.system(cmd)

        print('Template downloaded as ' + dest_file + '\n')

    except:
        print('\nPS1 template search failed!\n')


##################################




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
outdir = 'PSF_output_'+str(len(glob.glob('PSF_phot_*')))
if not os.path.exists(outdir): os.makedirs(outdir)

# A file to write final magnitudes
outFile = open('PSF_phot_'+str(len(glob.glob('PSF_phot_*')))+'.txt','w')
outFile.write('#image\tfilter\tmjd\tPSFmag\terr\tAPmag\terr\tZP\terr\tcomments')



# Prepare to plot images and PSFs

plt.figure(1,(14,7))
plt.ion()
plt.show()



################################
# Part one: get sequence stars
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


# Search for sequence star file (RA, dec, mags)

suggSeq = glob.glob('*seq.txt')

if len(suggSeq)==0:
    suggSeq = glob.glob('../*seq.txt')

if len(suggSeq)==0:
    suggSeq = glob.glob('../../*seq.txt')

if len(suggSeq)>0:
    seqFile = suggSeq[0]
else:
    print('No sequence star data found locally...')
    PS1catalog(RAdec[0],RAdec[1],magmin,magmax)
    seqFile = 'PS1_seq.txt'
    # except:
    #     sys.exit('Error: no sequence stars (*_seq.txt) found')

print('\n####################\n\nSequence star magnitudes found: '+seqFile)


seqDat = np.genfromtxt(seqFile,skip_header=1)

seqHead = np.genfromtxt(seqFile,skip_footer=len(seqDat),dtype=str)

seqMags = {}

for i in range(len(seqHead)-2):
    seqMags[seqHead[i+2]] = seqDat[:,i+2]


#### BEGIN LOOP OVER IMAGES ####


x_sh_1 = 0
y_sh_1 = 0

for image in ims:

    comment1 = ''


################################
# Part two: image header info
################################



    print('\nFile: ' + image)
    try:
        filtername = astropy.io.fits.getval(image,'FILTER')
    except:
        try:
            filtername = astropy.io.fits.getval(image,'FILTER1')
            if filtername == ('air' or 'none' or 'clear'):
                filtername = astropy.io.fits.getval(image,'FILTER2')
            if filtername == ('air' or 'none' or 'clear'):
                filtername = astropy.io.fits.getval(image,'FILTER3')
        except:
            try:
                filtername = astropy.io.fits.getval(image,'NCFLTNM2')
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
        mjd = astropy.io.fits.getval(image,'MJD')
    except:
        try:
            mjd = astropy.io.fits.getval(image,'MJD-OBS')
        except:
            try:
                jd = astropy.io.fits.getval(image,'JD')
                mjd = jd - 2400000
            except:
                mjd = 99999

        mjd = float(mjd)


#################################
# Part three: do some photometry
#################################



    print('\n#########\n'+filtername+'-band\n#########\n')


    im = astropy.io.fits.open(image)

    try:
        im[0].verify('fix')

        data = im[0].data
        header = im[0].header

    except:
        im[1].verify('fix')

        data = im[1].data
        header = im[1].header


    # background subtraction:

    bkg = photutils.background.Background2D(data,box_size=bkgbox)

    data -= bkg.background

########## plot data

    plt.clf()

    plt.subplots_adjust(left=0.05,right=0.99,top=0.99,bottom=-0.05)


    ax1 = plt.subplot2grid((2,4),(0,0),colspan=2,rowspan=2)


    ax1.imshow(data, origin='lower',cmap='gray',
                vmin=visualization.ZScaleInterval().get_limits(data)[0],
                vmax=visualization.ZScaleInterval().get_limits(data)[1])

    ax1.set_title(image)

    ax1.set_xlim(0,len(data))
    ax1.set_ylim(0,len(data))

    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)

    co = astropy.wcs.WCS(header=header).all_world2pix(seqDat[:,0],seqDat[:,1],1)
    co = np.array(list(zip(co[0],co[1])))

    # Remove any stars falling outside the image
    inframe = (co[:,0]>0)&(co[:,0]<len(data[0])-1)&(co[:,1]>0)&(co[:,1]<len(data)-1)
    co = co[inframe]

    orig_co = co.copy()

    # Mark sequence stars
    ax1.errorbar(co[:,0],co[:,1],fmt='s',mfc='none',markeredgecolor='C0',
                    markersize=8,markeredgewidth=1.5,
                    zorder=6)


    SNco = astropy.wcs.WCS(header=header).all_world2pix(RAdec[0],RAdec[1],0)

    ax1.errorbar(SNco[0],SNco[1],fmt='o',markeredgecolor='r',mfc='none',
                    markeredgewidth=3,markersize=20)


    plt.draw()



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

    del_x = np.mean(co[:,0]-orig_co[:,0])
    del_y = np.mean(co[:,1]-orig_co[:,1])

    for j in range(len(co)):
        apcircle = Circle((co[j,0], co[j,1]), aprad, facecolor='none',
                edgecolor='b', linewidth=1, alpha=1)
        ax1.add_patch(apcircle)

        ax1.text(co[j,0]+20,co[j,1]-20,str(j+1),color='cornflowerblue')



    # Define apertures and do simple photometry on sequence stars

    photaps = photutils.CircularAperture(co, r=aprad)

    skyaps = photutils.CircularAnnulus(co, r_in=aprad, r_out=aprad+skyrad)

    skymasks = skyaps.to_mask(method='center')

    # Get median sky in annulus for each star

    bkg_median = []
    for mask in skymasks:
        try:
            skydata = mask.multiply(data)
            skydata_1d = skydata[mask.data > 0]
            meansky, mediansky, sigsky = astropy.stats.sigma_clipped_stats(skydata_1d)
            bkg_median.append(mediansky)
        except:
            bkg_median.append(0.)
    bkg_median = np.array(bkg_median)
    photTab = photutils.aperture_photometry(data, photaps)
    photTab['local_sky'] = bkg_median
    photTab['aper_sum_sub'] = photTab['aperture_sum'] - bkg_median * photaps.area()

    # PSF Photometry

    # Required formats for photutils:
    nddata = astropy.nddata.NDData(data=data)
    psfinput = astropy.table.Table()
    psfinput['x'] = co[:,0]
    psfinput['y'] = co[:,1]

    # Create model from sequence stars
    happy = 'n'
    while happy!='y':

        # extract stars from image
        psfstars = photutils.psf.extract_stars(nddata, psfinput, size=aprad+skyrad)

        # build PSF

        print('Fitting PSF:')
        epsf_builder = photutils.EPSFBuilder(oversampling=1.0)
        epsf, fitted_stars = epsf_builder(psfstars)

        psf = epsf.data

        ax2 = plt.subplot2grid((2,4),(0,2))

        ax2.imshow(psf, origin='lower',cmap='gray',
                    vmin=visualization.ZScaleInterval().get_limits(psf)[0],
                    vmax=visualization.ZScaleInterval().get_limits(psf)[1])

        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)

        ax2.set_title('PSF')

        plt.draw()


        ax3 = plt.subplot2grid((2,4),(0,3),projection='3d')

        tmpArr = range(len(psf))

        X, Y = np.meshgrid(tmpArr,tmpArr)

        ax3.plot_surface(X,Y,psf,rstride=1,cstride=1,cmap='hot',alpha=0.5)

        ax3.set_zlim(np.min(psf),np.max(psf)*1.1)

        ax3.set_axis_off()


        plt.draw()


        happy = input('\nProceed with this PSF? [y] ')
        if not happy: happy = 'y'
        if happy != 'y':
            aprad = int(input('Try new aperture radius: [' +str(aprad)+']'))


    psfcoordTable = astropy.table.Table()

    # psfcoordTable['id'] = np.arange(len(co))
    psfcoordTable['x_0'] = co[:,0]
    psfcoordTable['y_0'] = co[:,1]
    psfcoordTable['flux_0'] = photTab['aper_sum_sub']

    grouper = photutils.psf.DAOGroup(crit_separation=10.)

    # need an odd number of pixels to fit PSF
    fitrad = 2*aprad - 1

    psfphot = photutils.psf.BasicPSFPhotometry(group_maker=grouper,
                    bkg_estimator=photutils.background.MMMBackground(),
                    psf_model=epsf, fitshape=fitrad,
                    finder=None, aperture_radius=aprad)

    psfphotTab = psfphot.do_photometry(data, init_guesses=psfcoordTable)

    psfsubIm = psfphot.get_residual_image()

    ax1.imshow(psfsubIm, origin='lower',cmap='gray',
                vmin=visualization.ZScaleInterval().get_limits(data)[0],
                vmax=visualization.ZScaleInterval().get_limits(data)[1])

    goodStars = (psfphotTab['flux_fit']/psfphotTab['flux_0']>0.95)&(psfphotTab['flux_fit']/psfphotTab['flux_0']<1.05)

    ax1.errorbar(co[:,0][~goodStars],co[:,1][~goodStars],fmt='x',mfc='none',
                markeredgewidth=2, color='C1',
                markersize=8,label='Rejected stars')

    ax1.text(len(data[:,0])*0.8,len(data[:,1])*0.9,'Rejected',color='C1')



########## Zero point from seq stars

    if filtername in seqMags:

        flux = np.array(psfphotTab['flux_fit'])[goodStars]
        seqIm = -2.5*np.log10(flux)

        zpList1 = seqMags[filtername][inframe][goodStars]-seqIm

        zp1 = np.mean(zpList1)
        errzp1 = np.std(zpList1)

        print('Initial zeropoint =  %.3f +/- %.3f\n' %(zp1,errzp1))

        checkMags = np.abs(seqIm+zp1-seqMags[filtername][inframe][goodStars])<errzp1*sigClip

        if False in checkMags:
            print('Rejecting stars from ZP: ')
            print(psfphotTab['id'][goodStars][~checkMags])
            print('ZPs:')
            print(seqMags[filtername][inframe][goodStars][~checkMags]-seqIm[~checkMags])

        ax1.errorbar(co[:,0][goodStars][~checkMags],
                    co[:,1][goodStars][~checkMags],fmt='x',mfc='none',
                    markeredgewidth=2, color='C3',
                    markersize=8,label='Rejected stars')

        zpList = seqMags[filtername][inframe][goodStars][checkMags]-seqIm[checkMags]

        ZP = np.mean(zpList)
        errZP = np.std(zpList)

        print('Zeropoint = %.3f +/- %.3f\n' %(ZP,errZP))

    else:
        ZP = np.nan
        errZP = np.nan

########### SN photometry

    SNco[0] += del_x
    SNco[1] += del_y

    SNco[0],SNco[1] = photutils.centroids.centroid_sources(data,SNco[0],SNco[1],
                                centroid_func=photutils.centroids.centroid_2dg)


    ax4 = plt.subplot2grid((2,4),(1,2))


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

    skycircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
            edgecolor='r', linewidth=3, alpha=1)
    ax4.add_patch(skycircle)

    plt.draw()



    # aperture

    photap = photutils.CircularAperture(SNco, r=aprad)

    skyap = photutils.CircularAnnulus(SNco, r_in=aprad, r_out=aprad+skyrad)

    skymask = skyap.to_mask(method='center')

    # Get median sky in annulus around transient
    bkg_median = []
    skydata = skymask[0].multiply(data)
    skydata_1d = skydata[skymask[0].data > 0]
    meansky, mediansky, sigsky = astropy.stats.sigma_clipped_stats(skydata_1d)
    bkg_median.append(mediansky)
    bkg_median = np.array(bkg_median)
    SNphotTab = photutils.aperture_photometry(data, photap)
    SNphotTab['local_sky'] = bkg_median
    SNphotTab['aper_sum_sub'] = SNphotTab['aperture_sum'] - bkg_median * photap.area()



    SNcoordTable = astropy.table.Table()
    SNcoordTable['x_0'] = SNco[0]
    SNcoordTable['y_0'] = SNco[1]
    SNcoordTable['flux_0'] = SNphotTab['aper_sum_sub']


    # psfphot = photutils.psf.BasicPSFPhotometry(group_maker=grouper,
    #                 bkg_estimator=photutils.background.MMMBackground(),
    #                 psf_model=epsf, fitshape=fitrad,
    #                 finder=None, aperture_radius=aprad)

    SNpsfphotTab = psfphot.do_photometry(data, init_guesses=SNcoordTable)

    SNpsfsubIm = psfphot.get_residual_image()

    ax5 = plt.subplot2grid((2,4),(1,3))

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

    skycircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
            edgecolor='r', linewidth=3, alpha=1)
    ax5.add_patch(skycircle)

    plt.draw()


    # Convert flux to instrumental magnitudes

    SNap = -2.5*np.log10(SNphotTab['aper_sum_sub'])
    # aperture mag error assuming Poisson noise
    errSNap = abs(SNap * np.sqrt( 1/SNphotTab['aper_sum_sub'] + 1/(bkg_median * photap.area()) ) )

    try:
        SNpsf = -2.5*np.log10(SNpsfphotTab['flux_fit'])
        errSNpsf = abs(SNpsf * SNpsfphotTab['flux_unc']/SNpsfphotTab['flux_fit'])
    except:
        SNpsf = np.nan
        errSNpsf = np.nan
        print('PSF fit failed, aperture mag only!')

    print('\n')

    if filtername in seqMags:

        calMagPsf = SNpsf + ZP

        errMagPsf = np.sqrt(errSNpsf**2 + errZP**2)


        calMagAp = SNap + ZP

        errMagAp = np.sqrt(errSNap**2 + errZP**2)

    else:
        calMagPsf = SNpsf

        errMagPsf = errSNpsf


        calMagAp = SNap

        errMagAp = errSNap

        comment1 = 'No ZP - instrumental mag only'

        print('> No ZP - instrumental mag only!!!')



    print('> PSF mag = '+'%.2f +/- %.2f' %(calMagPsf,errMagPsf))
    print('> Aperture mag = '+'%.2f +/- %.2f' %(calMagAp,errMagAp))


    comment = input('\n> Add comment to output file: ')

    if comment1:
        comment += (' // '+comment1)

    outFile.write('\n'+image+'\t'+filtername+'\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t'
                    %(mjd,calMagPsf,errMagPsf,calMagAp,errMagAp,ZP,errZP))
    outFile.write(comment)


outFile.close()


print('\n##########################################\nFinished!\nCalibrated PSF phot saved to ./PSF_phot.txt\nAperture photometry saved to ./ap_phot.txt\nCheck PSF_output/ for additional info\n##########################################')
