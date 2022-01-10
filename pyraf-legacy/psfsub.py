#!/usr/bin/env python

import numpy as np
import glob
from pyraf import iraf
from iraf import daophot
try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits
import sys
import shutil
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
import argparse
from matplotlib.patches import Circle
import requests
import time
try:
    from queryPS1 import PS1catalog
except:
    print 'Warning: PS1 MAST query package not found, must have sequence star data locally\n'
try:
    from queryPS1 import PS1cutouts
except:
    print 'Warning: PS1 image query package not found, must have template images locally\n'


####### Parameters to vary if things aren't going well: #####################
#
# aprad = 10      # aperture radius for phot
# nPSF = 10              # number of PSF stars to try
# recen_rad_stars = 10     # recentering radius: increase if centroid not found; decrease if wrong centroid found!
# recen_rad_sn = 5
# varOrd = 0             # PSF model order: -1 = pure analytic, 2 = high-order empirical corrections, 0/1 intermediate
# sigClip = 1         # Reject sequence stars if calculated ZP differs by this may sigma from the mean
#
#############


for i in glob.glob('*.mag.*'):
    os.remove(i)

for i in glob.glob('*.als.*'):
    os.remove(i)

for i in glob.glob('*.arj.*'):
    os.remove(i)

for i in glob.glob('*.sub.*'):
    os.remove(i)

for i in glob.glob('*.pst.*'):
    os.remove(i)

for i in glob.glob('*psf.*'):
    os.remove(i)

for i in glob.glob('*.psg.*'):
    os.remove(i)

for i in glob.glob('*_seqMags.txt'):
    os.remove(i)

for i in glob.glob('*pix*fits'):
    os.remove(i)

for i in glob.glob('*_psf_stars.txt'):
    os.remove(i)

for i in glob.glob('refcoords.txt'):
    os.remove(i)

for i in glob.glob('imagelist.txt'):
    os.remove(i)

for i in glob.glob('comlist.txt'):
    os.remove(i)

for i in glob.glob('shifts.txt'):
    os.remove(i)

for i in glob.glob('shifted_*'):
    os.remove(i)

for i in glob.glob('template0.fits'):
    os.remove(i)


parser = argparse.ArgumentParser()


parser.add_argument('--im','-i', dest='image', default='',
                    help='Image to reduce. One image at a time!')

parser.add_argument('--tmp','-t', dest='template', default='',
                    help='Template to subtract. One at a time!')

parser.add_argument('--keep','-k', dest='keep', default=False, action='store_true',
                    help='keep some intermediate images: aligned template and '
                    'pedestal images')

parser.add_argument('--ap', dest='aprad', default=10, type=int,
                    help='Radius for aperture/PSF phot.')

parser.add_argument('--sky', dest='skyrad', default=10, type=int,
                    help='Width of annulus for sky background.')

parser.add_argument('--npsf', dest='nPSF', default=10, type=int,
                    help='Number of PSF stars.')

parser.add_argument('--re', dest='recen_rad_stars', default=5, type=int,
                    help='Radius for recentering on stars.')

parser.add_argument('--resn', dest='recen_rad_sn', default=5, type=int,
                    help='Radius for recentering on SN.')

parser.add_argument('--var', dest='varOrd', default=0, type=int,
                    help='Order for PSF model.')

parser.add_argument('--sig', dest='sigClip', default=1, type=int,
                    help='Sigma clipping for rejecting sequence stars.')

parser.add_argument('--high', dest='z2', default=1, type=float,
                    help='Colour scaling for zoomed images; upper bound is '
                    'this value times the standard deviation of the counts.')

parser.add_argument('--low', dest='z1', default=1, type=float,
                    help='Colour scaling for zoomed images; lower bound is '
                    'this value times the standard deviation of the counts.')

parser.add_argument('--keepsub', dest='keep_sub', default=False, action='store_true',
                    help='Do not delete residual images during clean-up ')

parser.add_argument('--magmin', dest='magmin', default=21.5, type=float,
                    help='Faintest sequence stars to return from PS1 query ')

parser.add_argument('--magmax', dest='magmax', default=16.5, type=float,
                    help='Brightest sequence stars to return from PS1 query ')

parser.add_argument('--templatesize', dest='templatesize', default=2500, type=int,
                    help='Size of PS1 template to download (pixels) ')

parser.add_argument('--shifts', dest='shifts', default=False, action='store_true',
                    help='Apply manual shifts if WCS is a bit off ')

parser.add_argument('--trim', dest='trim', default=1, type=float,
                    help='Cut down target image by this factor in x and y ')



args = parser.parse_args()


image = args.image
template = args.template
keep = args.keep
shifts = args.shifts
trim = args.trim

aprad = args.aprad
skyrad = args.skyrad
nPSF = args.nPSF
recen_rad_stars = args.recen_rad_stars
recen_rad_sn = args.recen_rad_sn
varOrd = args.varOrd
sigClip = args.sigClip
z1 = args.z1
z2 = args.z2
magmin = args.magmin
magmax = args.magmax
templatesize = args.templatesize


##################################################

iraf.centerpars.calgo='centroid'
iraf.centerpars.cmaxiter=10
iraf.centerpars.cbox=recen_rad_stars
iraf.fitskypars.salgo='centroid'
iraf.fitskypars.annulus=aprad
iraf.fitskypars.dannulus=skyrad
iraf.photpars.apertures=aprad
iraf.photpars.zmag=0
iraf.datapars.sigma='INDEF'
iraf.datapars.datamin='INDEF'
iraf.datapars.datamax='INDEF'

daophot.daopars.function='gauss'
daophot.daopars.psfrad=aprad
daophot.daopars.fitrad=10
daophot.daopars.matchrad=3
daophot.daopars.sannu=max(5,aprad)
daophot.daopars.wsann=skyrad
daophot.daopars.varorder=varOrd
daophot.datapars.datamin='INDEF'
daophot.datapars.datamax='INDEF'
daophot.daopars.recenter='yes'
daophot.daopars.groupsky='yes'
daophot.daopars.fitsky='yes'


##################################################

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




################################
# Part one: Initial setup and checks
################################



outdir = 'PSF_sub_output_'+str(int(time.time()))

if not os.path.exists(outdir): os.makedirs(outdir)


outFile = open('PSF_sub_phot_'+str(int(time.time()))+'.txt','w')

ZPfile = open(os.path.join(outdir,'zeropoints.txt'),'w')


outFile.write('#image\tfilter\tmjd\tPSFmag\terr\tAPmag\terr\tcomments')




fig1 = plt.figure(1,(10,6))

plt.clf()

plt.ion()

plt.show()


# Get transient coordinates
suggSn = glob.glob('*coords.txt')

if len(suggSn)==0:
    suggSn = glob.glob('../*coords.txt')

if len(suggSn)==0:
    suggSn = glob.glob('../../*coords.txt')

if len(suggSn)>0:
    snFile = suggSn[0]
else:
    sys.exit('Error: no SN coordinates (*_coords.txt) found')

print '\n####################\n\nSN coordinates found: '+snFile

RAdec = np.genfromtxt(snFile)


# Get star catalog
suggSeq = glob.glob('*seq.txt')

if len(suggSeq)==0:
    suggSeq = glob.glob('../*seq.txt')

if len(suggSeq)==0:
    suggSeq = glob.glob('../../*seq.txt')

if len(suggSeq)>0:
    seqFile = suggSeq[0]
else:
    print 'No sequence star data found locally...'
    PS1catalog(RAdec[0],RAdec[1],magmin,magmax)
    seqFile = 'PS1_seq.txt'
    # except:
    #     sys.exit('Error: no sequence stars (*_seq.txt) found')

print '\n####################\n\nSequence star magnitudes found: '+seqFile


seqDat = np.genfromtxt(seqFile,skip_header=1)

seqHead = np.genfromtxt(seqFile,skip_footer=len(seqDat),dtype=str)

np.savetxt('coords',seqDat[:,:2],fmt='%e',delimiter='  ')

seqMags = {}

for i in range(len(seqHead)-2):
    seqMags[seqHead[i+2]] = seqDat[:,i+2]



# Get epoch

try:
    mjd = pyfits.getval(image,'MJD')
except:
    try:
        mjd = pyfits.getval(image,'MJD-OBS')
    except:
        try:
            jd = pyfits.getval(image,'JD')
            mjd = jd - 2400000
        except:
            mjd = 99999

    mjd = float(mjd)


# Check filters match

# image
print '\nFile: ' + image
try:
    filtername1 = pyfits.getval(image,'FILTER')
except:
    try:
        filtername1 = pyfits.getval(image,'FILTER1')
        if filtername1 == ('air' or 'none' or 'clear'):
            filtername1 = pyfits.getval(image,'FILTER2')
        if filtername1 == ('air' or 'none' or 'clear'):
            filtername1 = pyfits.getval(image,'FILTER3')
    except:
        try:
            filtername1 = pyfits.getval(image,'NCFLTNM2')
        except:
            filtername1 = 'none'
print 'Filter found in header: ' + filtername1
if filtername1=='none':
    filtername1 = raw_input('\n> Please enter filter ('+filtAll+') ')

for j in filtSyn:
    if filtername1 in filtSyn[j]:
        filtername1 = j
        print 'Standard filter = ' + filtername1

if filtername1 not in filtAll:
    filtername1 = raw_input('\n> Please enter filter ('+filtAll+') ')



# template
if not template:
    suggTemp = glob.glob(filtername1+'_template.fits')

    if len(suggTemp)==0:
        suggTemp = glob.glob('../'+filtername1+'_template.fits')

    if len(suggTemp)==0:
        suggTemp = glob.glob('../../'+filtername1+'_template.fits')

    if len(suggTemp)>0:
        template = suggTemp[0]
    else:
        sys.exit('No template found locally...')

    print '\n####################\n\nTemplate found: '+template


######## DOWNLOAD TEMPLATE FROM PS1 ##########

if template == 'ps1':
    if os.path.exists(filtername1+'_template.fits'):
        template = filtername1 + '_template.fits'
        filtername2 = filtername1
        print('PS1 template found in working directory')
    else:
        try:
            PS1cutouts(RAdec[0],RAdec[1],filtername1,templatesize)
            template = filtername1 + '_template.fits'
            filtername2 = filtername1
        except:
            sys.exit('Error: could not match template from PS1')


##############################################

else:
    print '\nFile: ' + template
    try:
        filtername2 = pyfits.getval(template,'FILTER')
    except:
        try:
            filtername2 = pyfits.getval(template,'FILTER1')
            if filtername2 == ('air' or 'none' or 'clear'):
                filtername2 = pyfits.getval(template,'FILTER2')
            if filtername2 == ('air' or 'none' or 'clear'):
                filtername2 = pyfits.getval(template,'FILTER3')
        except:
            try:
                filtername2 = pyfits.getval(template,'NCFLTNM2')
            except:
                try:
                    filtername2 = pyfits.getval(template,'HIERARCH FPA.FILTER')
                except:
                    filtername2 = 'none'
    print 'Filter found in header: ' + filtername2
    if filtername2=='none':
        filtername2 = raw_input('\n> Please enter filter ('+filtAll+') ')

    for j in filtSyn:
        if filtername2 in filtSyn[j]:
            filtername2 = j
            print 'Standard filter = ' + filtername2

    if filtername2 not in filtAll:
        filtername2 = raw_input('\n> Please enter filter ('+filtAll+') ')

    if filtername2!=filtername1:
        message = raw_input('\n!!!Warning: filters do not appear to match!!!')




for i in glob.glob(filtername2+'_aligned_*'):
    os.remove(i)

for i in glob.glob(image.split('.fits')[0]+'_ped.fits'):
    os.remove(i)

if trim>1.01:
    try:
        xlen = pyfits.getval(image,'NAXIS1')
        ylen = pyfits.getval(image,'NAXIS2')
    except:
        try:
            im = pyfits.open(image)
            imdata = im[0].data
        except:
            im = pyfits.open(image)
            imdata = im[1].data
        xlen = imdata.shape[0]
        ylen = imdata.shape[1]
    xtrim1 = int(xlen/2.-0.5*xlen/trim)
    xtrim2 = int(xlen/2.+0.5*xlen/trim)
    ytrim1 = int(ylen/2.-0.5*ylen/trim)
    ytrim2 = int(ylen/2.+0.5*ylen/trim)
    trimsection = '['+str(xtrim1)+':'+str(xtrim2)+','+str(ytrim1)+':'+str(ytrim2)+']'
    for i in glob.glob(image.split('.fits')[0]+'_trim.fits'):
        os.remove(i)
    iraf.imcopy(input=image+trimsection,output=image.split('.fits')[0]+'_trim.fits')
    image = image.split('.fits')[0]+'_trim.fits'

################################
# Part two: Display images and get coordinates
################################


### IMAGE ###
# Correct WCS keywords in header

ax1 = plt.subplot2grid((2,4),(0,0),colspan=2,rowspan=2)

try:
    im = pyfits.open(image)

    im[0].verify('fix')

    imdata = im[0].data

    imheader = im[0].header


except:
    im = pyfits.open(image)

    im[1].verify('fix')

    imdata = im[1].data

    imheader = im[1].header

### Correct for quirks of coordinate systems!!!

iraf.wcsreset(image=image,wcs='physical')

iraf.hedit(image=image,fields='TRIMSEC',value='[1:'+str(pyfits.getval(image,'NAXIS1'))+',1:'+str(pyfits.getval(image,'NAXIS2'))+']',verify='no')
iraf.hedit(image=image,fields='DATASEC',value='[1:'+str(pyfits.getval(image,'NAXIS1'))+',1:'+str(pyfits.getval(image,'NAXIS2'))+']',verify='no')



imdata[np.isnan(imdata)] = np.median(imdata[~np.isnan(imdata)])

ax1.imshow(imdata, origin='lower',cmap='gray',
                        vmin=np.mean(imdata[len(imdata)/2/2:3*len(imdata)/2/2,
                            len(imdata)/2/2:3*len(imdata)/2/2])-
                            z1*np.std(imdata[len(imdata)/2/2:3*len(imdata)/2/2,
                            len(imdata)/2/2:3*len(imdata)/2/2])*0.5,
                        vmax=np.mean(imdata[len(imdata)/2/2:3*len(imdata)/2/2,
                            len(imdata)/2/2:3*len(imdata)/2/2])+
                            z2*np.std(imdata[len(imdata)/2/2:3*len(imdata)/2/2,
                            len(imdata)/2/2:3*len(imdata)/2/2]))

ax1.set_title('Science image')


ax1.set_xlim(0,len(imdata))

ax1.set_ylim(0,len(imdata))

ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)

plt.draw()

iraf.wcsctran(input='coords',output=image+'_pix.fits',image=image,
                inwcs='world',outwcs='logical')

co = np.genfromtxt(image+'_pix.fits')

ax1.errorbar(co[:,0],co[:,1],fmt='o',mfc='none',markeredgewidth=3,
                markersize=20,label='Stars')

for q in range(len(co)):
   ax1.text(co[q,0]+20,co[q,1]-20,str(q+1),color='cornflowerblue')



iraf.wcsctran(input=snFile,output=image+'_SNpix.fits',image=image,
                inwcs='world',outwcs='logical')

SNco = np.genfromtxt(image+'_SNpix.fits')

ax1.errorbar(SNco[0],SNco[1],fmt='o',markeredgecolor='r',mfc='none',
                markeredgewidth=5,markersize=25)

ax1.text(SNco[0]+30,SNco[1]+30,'Target',color='r')


########### Centering

shutil.copy(image+'_pix.fits',image+'_orig_pix.fits')

pix_coords = np.genfromtxt(image+'_pix.fits')

# Manual shifts:
if shifts:
    x_sh = raw_input('\n> Add approx pixel shift in x? ['+str(x_sh_1)+']  ')
    if not x_sh: x_sh = x_sh_1
    x_sh = int(x_sh)
    x_sh_1 = x_sh

    y_sh = raw_input('\n> Add approx pixel shift in y? ['+str(y_sh_1)+']  ')
    if not y_sh: y_sh = y_sh_1
    y_sh = int(y_sh)
    y_sh_1 = y_sh

    pix_coords[:,0] += x_sh
    pix_coords[:,1] += y_sh

    np.savetxt(image+'_pix.fits',pix_coords)

# recenter on seq stars and generate star list for daophot:
iraf.phot(image=image,coords=image+'_pix.fits',output='default',
                interactive='no',verify='no',wcsin='logical',verbose='yes',
                Stdout='im_seq_phot1.txt')


imcoords = np.genfromtxt('im_seq_phot1.txt')

ax1.errorbar(imcoords[:,1],imcoords[:,2],fmt='s',mfc='none',markeredgecolor='b',
                markersize=10,markeredgewidth=1.5,label='Recentered',
                zorder=6)


orig_co = np.genfromtxt(image+'_orig_pix.fits')

del_x = np.mean(imcoords[:,1]-orig_co[:,0])
del_y = np.mean(imcoords[:,2]-orig_co[:,1])

daophot.pstselect(image=image,photfile='default',pstfile='default',
                    maxnpsf=nPSF,verify='no')



happy = 'n'
j = 1
rmStar = 999
while happy!='y':

    daophot.pselect(infiles=image+'.pst.'+str(j),
                    outfiles=image+'.pst.'+str(j+1),
                    expr='ID!='+str(rmStar))

    daophot.psf(image=image,photfile='default',pstfile='default',
                psfimage='default',opstfile='default',groupfil='default',
                verify='no',interactive='no')

    iraf.txdump(textfile=image+'.pst.'+str(j+1),fields='ID',expr='yes',
                Stdout=os.path.join(outdir,image+'_psf_stars.txt'))

    iraf.txdump(textfile=image+'.pst.'+str(j+1),fields='XCENTER,YCENTER',
                expr='yes',Stdout=os.path.join(outdir,image+'_psf_coords.txt'))

    psfList = np.genfromtxt(os.path.join(outdir,image+'_psf_stars.txt'),
                            dtype='int')

    for p in psfList:
        ax1.errorbar(imcoords[p-1,1],imcoords[p-1,2],fmt='*',mfc='none',
                        markeredgecolor='lime',markeredgewidth=2,
                        markersize=30,label='Used in PSF fit')

    # This doesn't remove stars from plot - need to give some thought!!!

    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(),
                numpoints=1,prop={'size':16}, handletextpad=0.5,
                labelspacing=0.5, borderaxespad=1, ncol=3,
                bbox_to_anchor=(1, 0.2))

    daophot.seepsf(psfimage=image+'.psf.'+str(j)+'.fits',
                    image=os.path.join(outdir,image+'_psf.fits'))

    psfIm = pyfits.open(os.path.join(outdir,image+'_psf.fits'))

    psfIm[0].verify('fix')

    psf = psfIm[0].data


    ax2 = plt.subplot2grid((2,4),(0,2))

    ax2.imshow(psf, origin='lower',cmap='gray',
                vmin=np.mean(psf)-np.std(psf)*1.,
                vmax=np.mean(psf)+np.std(psf)*3.)

    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)

    ax2.set_title('PSF')


    ax3 = plt.subplot2grid((2,4),(0,3),projection='3d')

    tmpArr = range(len(psf))

    X, Y = np.meshgrid(tmpArr,tmpArr)

    ax3.plot_surface(X,Y,psf,rstride=1,cstride=1,cmap='hot',alpha=0.5)

    ax3.set_zlim(np.min(psf),np.max(psf)*1.1)

    ax3.set_axis_off()

    plt.draw()


    happy = raw_input('\n> Happy with PSF? [y]')

    if not happy: happy = 'y'

    if happy != 'y':
        rmStar = raw_input('\n> Star to remove: ')
        j+=1
        os.remove(os.path.join(outdir,image+'_psf.fits'))


daophot.allstar(image=image,photfile='default',psfimage='default',
                allstarfile='default',rejfile='default',subimage='default',
                verify='no',verbose='yes',fitsky='yes')

sub1 = pyfits.open(image+'.sub.1.fits')

sub1[0].verify('fix')

sub0 = sub1[0].data

ax1.imshow(sub0, origin='lower',cmap='gray',
            vmin=np.mean(imdata[len(imdata)/2/2:3*len(imdata)/2/2,
                len(imdata)/2/2:3*len(imdata)/2/2])-
                z1*np.std(imdata[len(imdata)/2/2:3*len(imdata)/2/2,
                len(imdata)/2/2:3*len(imdata)/2/2])*0.5,
            vmax=np.mean(imdata[len(imdata)/2/2:3*len(imdata)/2/2,
                len(imdata)/2/2:3*len(imdata)/2/2])+
                z2*np.std(imdata[len(imdata)/2/2:3*len(imdata)/2/2,
                len(imdata)/2/2:3*len(imdata)/2/2]))

plt.draw()

# Mask stars with no PSF fit

iraf.txdump(textfile=image+'.als.1',fields='ID,XCENTER,YCENTER,MAG,MERR',
            expr='yes',Stdout=os.path.join(outdir,image+'_seqMags.txt'))

seqIm1 = np.genfromtxt(os.path.join(outdir,image+'_seqMags.txt'))
seqIm1 = seqIm1[seqIm1[:,0].argsort()]
immask = np.array(seqIm1[:,0],dtype=int)[~np.isnan(seqIm1[:,3])]-1

########## Zero point from seq stars

if filtername1 in seqMags:

    seqIm = seqIm1[:,3]
    seqErr = seqIm1[:,4]

    zpList1 = seqMags[filtername1][immask]-seqIm

    zp1 = np.mean(zpList1)
    errzp1 = np.std(zpList1)

    print 'Initial zeropoint =  %.3f +/- %.3f\n' %(zp1,errzp1)

    np.savetxt(os.path.join(outdir,'zp_list.txt'),zip(immask+1,zpList1),
    fmt='%d\t%.3f')

    checkMags = np.abs(seqIm+zp1-seqMags[filtername1][immask])<errzp1*sigClip

    print 'Rejecting stars from ZP: '
    print seqIm1[:,0][~checkMags]
    print 'ZPs:'
    print seqMags[filtername1][immask][~checkMags]-seqIm[~checkMags]

    zpList = seqMags[filtername1][immask][checkMags]-seqIm[checkMags]

    ZP = np.mean(zpList)
    errZP = np.std(zpList)

    print 'Zeropoint = %.3f +/- %.3f\n' %(ZP,errZP)

    ZPfile.write(image+'\t'+filtername1+'\t%.2f\t%.2f\t%.2f\n' %(mjd,ZP,errZP))

plt.draw()

# con = raw_input('\n> Enter to proceed to template...')

# plt.close(fig1)


### TEMPLATE ###

iraf.imcopy(input=template,output='template0.fits')

template = 'template0.fits'

# if pyfits.getval(template,'HIERARCH FPA.TELESCOPE')=='PS1':
    # print('\n\n> Image from PS1, applying RA fix\n')
try:
    pyfits.getval(template,'CD1_1')
except:
    print('\n> Outdated WCS keywords, fixing...')
    iraf.hedit(image=template,fields='CD1_1',value='(-1*CDELT1)',add='yes',
                addonly='yes',verify='no',update='yes')
    iraf.hedit(image=template,fields='CD2_2',value='(CDELT2)',add='yes',
                addonly='yes',verify='no',update='yes')
    iraf.hedit(image=template,fields='CD1_2',value='0',add='yes',
                addonly='yes',verify='no',update='yes')
    iraf.hedit(image=template,fields='CD2_1',value='0',add='yes',
                addonly='yes',verify='no',update='yes')
    iraf.hedit(image=template,fields='PC001001',add='no',addonly='no',
                delete='yes',verify='no',update='yes')
    iraf.hedit(image=template,fields='PC001002',add='no',addonly='no',
                delete='yes',verify='no',update='yes')
    iraf.hedit(image=template,fields='PC002001',add='no',addonly='no',
                delete='yes',verify='no',update='yes')
    iraf.hedit(image=template,fields='PC002002',add='no',addonly='no',
                delete='yes',verify='no',update='yes')

# fig2 = plt.figure(2,(10,6))

plt.clf()

plt.ion()

plt.show()

ax1 = plt.subplot2grid((2,4),(0,0),colspan=2,rowspan=2)

try:
    tmp = pyfits.open(template)

    tmp[0].verify('fix')

    tdata = tmp[0].data

    theader = tmp[0].header


except:
    tmp = pyfits.open(template)

    tmp[1].verify('fix')

    tdata = tmp[1].data

    theader = tmp[1].header


iraf.wcsreset(image=template,wcs='physical')

iraf.hedit(image=template,fields='TRIMSEC',value='[1:'+str(pyfits.getval(image,'NAXIS1'))+',1:'+str(pyfits.getval(image,'NAXIS2'))+']',verify='no')
iraf.hedit(image=template,fields='DATASEC',value='[1:'+str(pyfits.getval(image,'NAXIS1'))+',1:'+str(pyfits.getval(image,'NAXIS2'))+']',verify='no')


tdata[np.isnan(tdata)] = np.median(tdata[~np.isnan(tdata)])

ax1.imshow(tdata, origin='lower',cmap='gray',
                        vmin=np.mean(tdata[len(tdata)/2/2:3*len(tdata)/2/2,
                            len(tdata)/2/2:3*len(tdata)/2/2])-
                            z1*np.std(tdata[len(tdata)/2/2:3*len(tdata)/2/2,
                            len(tdata)/2/2:3*len(tdata)/2/2])*0.5,
                        vmax=np.mean(tdata[len(tdata)/2/2:3*len(tdata)/2/2,
                            len(tdata)/2/2:3*len(tdata)/2/2])+
                            z2*np.std(tdata[len(tdata)/2/2:3*len(tdata)/2/2,
                            len(tdata)/2/2:3*len(tdata)/2/2]))

ax1.set_title('Template image')


ax1.set_xlim(0,len(tdata))

ax1.set_ylim(0,len(tdata))

ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)

plt.draw()

iraf.wcsctran(input='coords',output=template+'_pix.fits',image=template,
                inwcs='world',outwcs='logical')

co = np.genfromtxt(template+'_pix.fits')

ax1.errorbar(co[:,0],co[:,1],fmt='o',mfc='none',markeredgewidth=3,
                markersize=20)

for q in range(len(co)):
   ax1.text(co[q,0]+20,co[q,1]-20,str(q+1),color='cornflowerblue')

plt.draw()

########### Centering

# Manual shifts:
if shifts:
    x_sh = raw_input('\n> Add approx pixel shift in x? [0]  ')
    if not x_sh: x_sh = 0
    x_sh = int(x_sh)

    y_sh = raw_input('\n> Add approx pixel shift in y? [0]  ')
    if not y_sh: y_sh = 0
    y_sh = int(y_sh)


    pix_coords = np.genfromtxt(template+'_pix.fits')
    pix_coords[:,0] += x_sh
    pix_coords[:,1] += y_sh

    np.savetxt(template+'_pix.fits',pix_coords)

# recenter on seq stars and generate star list for daophot:
iraf.phot(image=template,coords=template+'_pix.fits',output='default',
                interactive='no',verify='no',wcsin='logical',verbose='yes',
                Stdout='tmpl_seq_phot1.txt')


tcoords = np.genfromtxt('tmpl_seq_phot1.txt')

ax1.errorbar(tcoords[:,1],tcoords[:,2],fmt='s',mfc='none',markeredgecolor='b',
                markersize=10,markeredgewidth=1.5,label='Recentered',
                zorder=6)

plt.draw()

daophot.pstselect(image=template,photfile='default',pstfile='default',
                    maxnpsf=nPSF,verify='no')



happy = 'n'
k = 1
rmStar = 999
while happy!='y':

    daophot.pselect(infiles=template+'.pst.'+str(k),
                    outfiles=template+'.pst.'+str(k+1),
                    expr='ID!='+str(rmStar))

    daophot.psf(image=template,photfile='default',pstfile='default',
                psfimage='default',opstfile='default',groupfil='default',
                verify='no',interactive='no')

    iraf.txdump(textfile=template+'.pst.'+str(k+1),fields='ID',expr='yes',
                Stdout=os.path.join(outdir,template+'_psf_stars.txt'))

    psfList = np.genfromtxt(os.path.join(outdir,template+'_psf_stars.txt'),
                            dtype='int')

    for p in psfList:
        ax1.errorbar(tcoords[p-1,1],tcoords[p-1,2],fmt='*',mfc='none',
                        markeredgecolor='lime',markeredgewidth=2,
                        markersize=30,label='Used in PSF fit')

    # This doesn't remove stars from plot - need to give some thought!!!

    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(),
                numpoints=1,prop={'size':16}, handletextpad=0.5,
                labelspacing=0.5, borderaxespad=1, ncol=3,
                bbox_to_anchor=(1, 1.2))

    daophot.seepsf(psfimage=template+'.psf.'+str(k)+'.fits',
                    image=os.path.join(outdir,template+'_psf.fits'))

    psfIm = pyfits.open(os.path.join(outdir,template+'_psf.fits'))

    psfIm[0].verify('fix')

    psf = psfIm[0].data


    ax2 = plt.subplot2grid((2,4),(0,2))

    ax2.imshow(psf, origin='lower',cmap='gray',
                vmin=np.mean(psf)-np.std(psf)*1.,
                vmax=np.mean(psf)+np.std(psf)*3.)

    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)

    ax2.set_title('PSF')


    ax3 = plt.subplot2grid((2,4),(0,3),projection='3d')

    tmpArr = range(len(psf))

    X, Y = np.meshgrid(tmpArr,tmpArr)

    ax3.plot_surface(X,Y,psf,rstride=1,cstride=1,cmap='hot',alpha=0.5)

    ax3.set_zlim(np.min(psf),np.max(psf)*1.1)

    ax3.set_axis_off()

    plt.draw()


    happy = raw_input('\n> Happy with PSF? [y]')

    if not happy: happy = 'y'

    if happy != 'y':
        rmStar = raw_input('\n> Star to remove: ')
        k+=1
        os.remove(os.path.join(outdir,template+'_psf.fits'))


daophot.allstar(image=template,photfile='default',psfimage='default',
                allstarfile='default',rejfile='default',subimage='default',
                verify='no',verbose='yes',fitsky='yes')

sub1 = pyfits.open(template+'.sub.1.fits')

sub1[0].verify('fix')

sub0 = sub1[0].data

ax1.imshow(sub0, origin='lower',cmap='gray',
            vmin=np.mean(tdata[len(tdata)/2/2:3*len(tdata)/2/2,
                len(tdata)/2/2:3*len(tdata)/2/2])-
                z1*np.std(tdata[len(tdata)/2/2:3*len(tdata)/2/2,
                len(tdata)/2/2:3*len(tdata)/2/2])*0.5,
            vmax=np.mean(tdata[len(tdata)/2/2:3*len(tdata)/2/2,
                len(tdata)/2/2:3*len(tdata)/2/2])+
                z2*np.std(tdata[len(tdata)/2/2:3*len(tdata)/2/2,
                len(tdata)/2/2:3*len(tdata)/2/2]))


# Mask stars with no PSF fit

iraf.txdump(textfile=template+'.als.1',fields='ID,XCENTER,YCENTER,MAG,MERR',
            expr='yes',Stdout=os.path.join(outdir,template+'_seqMags.txt'))

tmplIm1 = np.genfromtxt(os.path.join(outdir,template+'_seqMags.txt'))
tmplIm1 = tmplIm1[tmplIm1[:,0].argsort()]
tmplmask = np.array(tmplIm1[:,0],dtype=int)[~np.isnan(tmplIm1[:,3])]-1

plt.draw()

# con = raw_input('\n> Enter to align images...')

################################
# Part three: Geomap and geotran
################################


goodStars = np.intersect1d(immask,tmplmask)+1

# Transform TEMPLATE to match IMAGE!

geoinput = []
for i in goodStars:
    geoinput.append([float(seqIm1[:,1][seqIm1[:,0]==i]),
                        float(seqIm1[:,2][seqIm1[:,0]==i]),
                        float(tmplIm1[:,1][tmplIm1[:,0]==i]),
                        float(tmplIm1[:,2][tmplIm1[:,0]==i])])

# geoinput=list(zip(imcoords[:,0],imcoords[:,1],tcoords[:,0],tcoords[:,1]))

np.savetxt('geomap_input.txt',geoinput,fmt='%.2f')


#
# try:
#     imagesec = pyfits.getval(image,'TRIMSEC').split('[')[1].split(']')[0]
#     xmin = imagesec.split(',')[0].split(':')[0]
#     xmax = imagesec.split(',')[0].split(':')[-1]
#     ymin = imagesec.split(',')[-1].split(':')[0]
#     ymax = imagesec.split(',')[-1].split(':')[-1]
# except:
#     try:
#         imagesec = pyfits.getval(image,'DATASEC').split('[')[1].split(']')[0]
#         xmin = imagesec.split(',')[0].split(':')[0]
#         xmax = imagesec.split(',')[0].split(':')[-1]
#         ymin = imagesec.split(',')[-1].split(':')[0]
#         ymax = imagesec.split(',')[-1].split(':')[-1]
#     except:

try:
    xmin = 1
    xmax = pyfits.getval(image,'NAXIS1')
    ymin = 1
    ymax = pyfits.getval(image,'NAXIS2')
except:
    xmin = 1
    xmax = imdata.shape[0]
    ymin = 1
    ymax = imdata.shape[1]


iraf.geomap(input='geomap_input.txt',database='geomap_output.txt',
            xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,
            fitgeom='general',interac='no')

iraf.geotran(input=template,output=filtername2+'_aligned_tmpl.fits',
            database='geomap_output.txt',transform='geomap_input.txt')

# pedestal = max(np.abs(np.min(tdata)),np.abs(np.min(imdata)))*1.1
#
# iraf.imarith(operand1=filtername2+'_aligned_tmpl.fits',op='+',
#                 operand2=pedestal,result=filtername2+'_aligned_tmpl_ped.fits')
#
# iraf.imarith(operand1=image,op='+',operand2=pedestal,
#                 result=image.split('.fits')[0]+'_ped.fits')

iraf.imcopy(input=filtername2+'_aligned_tmpl.fits',
            output=filtername2+'_aligned_tmpl_ped.fits')

iraf.imcopy(input=image,output=image.split('.fits')[0]+'_ped.fits')

iraf.imreplace(image=filtername2+'_aligned_tmpl_ped.fits',
                value=np.median(tdata)+0.001, upper=0.1)

iraf.imreplace(image=image.split('.fits')[0]+'_ped.fits',
                value=np.median(imdata)+0.001, upper=0.1)


# fig3 = plt.figure(3,(10,6))
plt.clf()

# NEED TO MEASURE PSF ON TRANSFORMED IMAGE!


ax1 = plt.subplot2grid((2,4),(0,0),colspan=2,rowspan=2)

# ax1 = plt.subplot(111)

try:
    ta = pyfits.open(filtername2+'_aligned_tmpl_ped.fits')

    ta[0].verify('fix')

    tadata = ta[0].data

    taheader = ta[0].header


except:
    ta = pyfits.open(filtername2+'_aligned_tmpl_ped.fits')

    ta[1].verify('fix')

    tadata = ta[1].data

    taheader = ta[1].header


ax1.imshow(tadata, origin='lower',cmap='gray',
                        vmin=np.mean(tadata[len(tadata)/2/2:3*len(tadata)/2/2,
                            len(tadata)/2/2:3*len(tadata)/2/2])-
                            z1*np.std(tadata[len(tadata)/2/2:3*len(tadata)/2/2,
                            len(tadata)/2/2:3*len(tadata)/2/2])*0.5,
                        vmax=np.mean(tadata[len(tadata)/2/2:3*len(tadata)/2/2,
                            len(tadata)/2/2:3*len(tadata)/2/2])+
                            z2*np.std(tadata[len(tadata)/2/2:3*len(tadata)/2/2,
                            len(tadata)/2/2:3*len(tadata)/2/2]))

ax1.set_title('Template aligned to image')


ax1.set_xlim(0,len(tadata))

ax1.set_ylim(0,len(tadata))

ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)

plt.draw()



ax1.errorbar(imcoords[:,1],imcoords[:,2],fmt='s',mfc='none',markeredgecolor='b',
                markersize=10,markeredgewidth=1.5,label='Recentered',
                zorder=6)

iraf.phot(image=filtername2+'_aligned_tmpl_ped.fits',
            coords=os.path.join(outdir,image+'_psf_coords.txt'),
            output='default',
            interactive='no',verify='no',wcsin='logical',verbose='yes',
            Stdout='im_seq_phot1.txt')

daophot.pstselect(image=filtername2+'_aligned_tmpl_ped.fits',photfile='default',
                    pstfile='default',maxnpsf=nPSF,verify='no')

daophot.psf(image=filtername2+'_aligned_tmpl_ped.fits',
            photfile=filtername2+'_aligned_tmpl_ped.fits.mag.1',
            pstfile=filtername2+'_aligned_tmpl_ped.fits.pst.1',
            psfimage='default',opstfile='default',
            groupfil='default',verify='no',interactive='no')


iraf.txdump(textfile=filtername2+'_aligned_tmpl_ped.fits.pst.2',
            fields='ID,XCENTER,YCENTER',expr='yes',
            Stdout=os.path.join(outdir,filtername2+'_aligned_tmpl_ped_psf_coords.txt'))

psfList = np.genfromtxt(os.path.join(outdir,filtername2+'_aligned_tmpl_ped_psf_coords.txt'))

ax1.errorbar(psfList[:,1],psfList[:,2],fmt='*',mfc='none',
                    markeredgecolor='lime',markeredgewidth=2,
                    markersize=30,label='Used in PSF fit')


# handles, labels = ax1.get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# ax1.legend(by_label.values(), by_label.keys(),
#             numpoints=1,prop={'size':16}, handletextpad=0.5,
#             labelspacing=0.5, borderaxespad=1, ncol=3,
#             bbox_to_anchor=(1, 1.2))

daophot.seepsf(psfimage=filtername2+'_aligned_tmpl_ped.fits.psf.1.fits',
                image=os.path.join(outdir,filtername2+'_aligned_tmpl_ped_psf.fits'))

psfTA = pyfits.open(os.path.join(outdir,filtername2+'_aligned_tmpl_ped_psf.fits'))

psfTA[0].verify('fix')

psf = psfTA[0].data


ax2 = plt.subplot2grid((2,4),(0,2))

ax2.imshow(psf, origin='lower',cmap='gray',
            vmin=np.mean(psf)-np.std(psf)*1.,
            vmax=np.mean(psf)+np.std(psf)*3.)

ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

ax2.set_title('PSF')


ax3 = plt.subplot2grid((2,4),(0,3),projection='3d')

tmpArr = range(len(psf))

X, Y = np.meshgrid(tmpArr,tmpArr)

ax3.plot_surface(X,Y,psf,rstride=1,cstride=1,cmap='hot',alpha=0.5)

ax3.set_zlim(np.min(psf),np.max(psf)*1.1)

ax3.set_axis_off()

plt.draw()

# con = raw_input('\n> Enter to subtract images...')

################################
# Part four: Run HOTPANTS
################################


inim = image.split('.fits')[0]+'_ped.fits'
tmplim = filtername2+'_aligned_tmpl_ped.fits'
subim = image.split('.fits')[0]+'_sub.fits'
convim = filtername2+'_aligned_tmpl_ped_conv.fits'
tu = int(np.max(tdata)/1.3)
iu = int(np.max(imdata)/1.3)

gainIm = 1.0
try:
    gainIm = pyfits.getval(image,'GAIN')
except:
    try:
        gainIm = pyfits.getval(image,'EGAIN')
    except:
        try:
            gainIm = pyfits.getval(image,'HIERARCH ESO DET OUT1 GAIN')
        except:
            gainIm = 1.0

noiseIm = 5.0
try:
    noiseIm = pyfits.getval(image,'READNOISE')
except:
    try:
        noiseIm = pyfits.getval(image,'READNOIS')
    except:
        try:
            noiseIm = pyfits.getval(image,'RDNOISE')
        except:
            try:
                noiseIm = pyfits.getval(image,'ENOISE')
            except:
                try:
                    noiseIm = pyfits.getval(image,'HIERARCH ESO DET OUT1 RON')
                except:
                    noiseIm = 5.0

gainTmpl = 1.0
try:
    gainTmpl = pyfits.getval(template,'GAIN')
except:
    try:
        gainTmpl = pyfits.getval(template,'EGAIN')
    except:
        try:
            gainTmpl = pyfits.getval(template,'HIERARCH ESO DET OUT1 GAIN')
        except:
            gainTmpl = 1.0

noiseTmpl = 5.0
try:
    noiseTmpl = pyfits.getval(template,'READNOISE')
except:
    try:
        noiseTmpl = pyfits.getval(template,'READNOIS')
    except:
        try:
            noiseTmpl = pyfits.getval(template,'RDNOISE')
        except:
            try:
                noiseTmpl = pyfits.getval(template,'ENOISE')
            except:
                try:
                    noiseTmpl = pyfits.getval(template,'HIERARCH ESO DET OUT1 RON')
                except:
                    noiseTmpl = 5.0


hwhmIm1 = pyfits.getval(image+'.psf.'+str(j)+'.fits','PAR1')
hwhmIm2 = pyfits.getval(image+'.psf.'+str(j)+'.fits','PAR2')
sigmaIm = 2./2.355 * (hwhmIm1+hwhmIm2)/2    # Average x and y direction; HWHM->sigma

hwhmTmpl1 = pyfits.getval(filtername2+'_aligned_tmpl_ped.fits.psf.1.fits','PAR1')
hwhmTmpl2 = pyfits.getval(filtername2+'_aligned_tmpl_ped.fits.psf.1.fits','PAR2')

# hwhmTmpl1_0 = pyfits.getval(template+'.psf.'+str(k)+'.fits','PAR1')
# hwhmTmpl2_0 = pyfits.getval(template+'.psf.'+str(k)+'.fits','PAR2')
#
# xscalelist = []
# yscalelist = []
#
# geooutput = open('geomap_output.txt','r')
#
# for line in geooutput.readlines():
#     if 'xmag' in line:
#         xscalelist.append(float(line.split('\t')[-1]))
#     if 'ymag' in line:
#         yscalelist.append(float(line.split('\t')[-1]))
#
# xscale = xscalelist[0]
# yscale = yscalelist[0]
#
# hwhmTmpl1 = hwhmTmpl1_0*xscale
# hwhmTmpl2 = hwhmTmpl2_0*yscale

sigmaTmpl = 2./2.355 * (hwhmTmpl1+hwhmTmpl2)/2

psfMatch = np.sqrt(sigmaIm**2 + sigmaTmpl**2)

gaussFlag = ' -ng 3 6 %.3f 4 %.3f 2 %.3f ' %(0.5*psfMatch, psfMatch, 2*psfMatch)


whichWay = 'n'

if sigmaTmpl > sigmaIm:
    whichWay = raw_input('!!!Warning: image sharper than template!!!\n'
                        'Do you want to convolve image instead (experimental)'
                        ' y/[n] ')
    if not whichWay: whichWay = 'n'

if whichWay == 'y':
    # Convolve image to template (experimental)
    os.system('hotpants -inim '+inim+' -tmplim '+tmplim+' -outim '+subim+
                ' -n i -c i -tu '+str(tu)+' -iu '+str(iu)+' -tg '+str(gainTmpl)+
                ' -tr '+str(noiseTmpl)+' -ig '+str(gainIm)+' -ir '+
                str(noiseTmpl)+gaussFlag+' -oci '+convim)
    label = 'Image convolved to template'
    psfModel = filtername2+'_aligned_tmpl_ped.fits.psf.1.fits'
else:
    # Convolve template to image (default)
    os.system('hotpants -inim '+inim+' -tmplim '+tmplim+' -outim '+subim+
                ' -n i -c t -tu '+str(tu)+' -iu '+str(iu)+' -tg '+str(gainTmpl)+
                ' -tr '+str(noiseTmpl)+' -ig '+str(gainIm)+' -ir '+
                str(noiseTmpl)+gaussFlag+' -oci '+convim)
    label = 'Template convolved to image'
    psfModel = image+'.psf.'+str(j)+'.fits'


# fig4 = plt.figure(4,(10,6))
plt.clf()

ax1 = plt.subplot(121)

try:
    diff = pyfits.open(subim)

    diff[0].verify('fix')

    dfdata = diff[0].data

    dfheader = diff[0].header


except:
    diff = pyfits.open(subim)

    diff[1].verify('fix')

    dfdata = diff[1].data

    dfheader = diff[1].header


dfdata[np.isnan(dfdata)] = np.median(dfdata[~np.isnan(dfdata)])

ax1.imshow(dfdata, origin='lower',cmap='gray',
                        vmin=np.mean(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2])-
                            z1*np.std(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2])*0.5,
                        vmax=np.mean(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2])+
                            z2*np.std(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2]))

ax1.set_title(label)


ax1.set_xlim(0,len(dfdata))

ax1.set_ylim(0,len(dfdata))

ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)

plt.draw()


################################
# Part five: Photometry on transient
################################


SNco[0] += del_x
SNco[1] += del_y

iraf.centerpars.cbox = recen_rad_sn

recen_rad_1 = recen_rad_sn

np.savetxt(image+'_SNpix_sh.fits',SNco.reshape(1,2),fmt='%.6f')

recen = 'y'
s = 1
while recen!='n':

    iraf.phot(image=image.split('.fits')[0]+'_sub.fits',
                coords=image+'_SNpix_sh.fits',output='default',
                interactive='no',verify='no',wcsin='logical',verbose='yes')

    daophot.allstar(image=image.split('.fits')[0]+'_sub.fits',
                    photfile=image.split('.fits')[0]+'_sub.fits.mag.'+str(s),
                    psfimage=psfModel,allstarfile='default',
                    rejfile='default',subimage='default',verify='no',
                    verbose='yes',fitsky='yes',recenter='no')

    sub1 = pyfits.open(image.split('.fits')[0]+'_sub.fits.sub.'+str(s)+'.fits')

    sub1[0].verify('fix')

    sub = sub1[0].data

    ax4 = plt.subplot2grid((2,4),(1,2))


    ax4.imshow(dfdata, origin='lower',cmap='gray',
                        vmin=z1*np.mean(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2])-
                            np.std(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2])*0.5,
                        vmax=z2*np.mean(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2])+
                            np.std(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2]))


    ax4.set_xlim(SNco[0]-(aprad+skyrad),SNco[0]+(aprad+skyrad))
    ax4.set_ylim(SNco[1]-(aprad+skyrad),SNco[1]+(aprad+skyrad))

    ax4.get_yaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)

    ax4.set_title('Target')

    skycircle = Circle((SNco[0], SNco[1]), max(5,aprad), facecolor='none',
            edgecolor='b', linewidth=3, alpha=1)
    ax4.add_patch(skycircle)

    skycircle2 = Circle((SNco[0], SNco[1]), max(5,aprad)+skyrad, facecolor='none',
            edgecolor='b', linewidth=3, alpha=1)
    ax4.add_patch(skycircle2)

    apcircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
            edgecolor='r', linewidth=3, alpha=1)
    ax4.add_patch(apcircle)


    ax5 = plt.subplot2grid((2,4),(1,3))

    ax5.imshow(sub, origin='lower',cmap='gray',
                        vmin=z1*np.mean(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2])-
                            np.std(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2])*0.5,
                        vmax=z2*np.mean(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2])+
                            np.std(dfdata[len(dfdata)/2/2:3*len(dfdata)/2/2,
                            len(dfdata)/2/2:3*len(dfdata)/2/2]))


    ax5.set_xlim(SNco[0]-(aprad+skyrad),SNco[0]+(aprad+skyrad))
    ax5.set_ylim(SNco[1]-(aprad+skyrad),SNco[1]+(aprad+skyrad))

    ax5.get_yaxis().set_visible(False)
    ax5.get_xaxis().set_visible(False)

    ax5.set_title('Subtracted image')

    skycircle = Circle((SNco[0], SNco[1]), max(5,aprad), facecolor='none',
            edgecolor='b', linewidth=3, alpha=1)
    ax5.add_patch(skycircle)

    skycircle2 = Circle((SNco[0], SNco[1]), max(5,aprad)+skyrad, facecolor='none',
            edgecolor='b', linewidth=3, alpha=1)
    ax5.add_patch(skycircle2)

    apcircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
            edgecolor='r', linewidth=3, alpha=1)
    ax5.add_patch(apcircle)


    plt.draw()


    recen = raw_input('\n> Adjust recentering radius? [n]  ')

    if not recen: recen = 'n'

    if recen!= 'n':
        recen_rad = raw_input('\n> Enter radius ['+str(recen_rad_1)+']  ')
        if not recen_rad: recen_rad = recen_rad_1
        recen_rad = int(recen_rad)
        iraf.centerpars.cbox = recen_rad
        recen_rad_1 = recen_rad
        s += 1


iraf.txdump(textfile=image.split('.fits')[0]+'_sub.fits.mag.'+str(s),
            fields='MAG,MERR',expr='yes',
            Stdout=os.path.join(outdir,image+'_SN_ap.txt'))

iraf.txdump(textfile=image.split('.fits')[0]+'_sub.fits.als.'+str(s),
            fields='MAG,MERR',expr='yes',
            Stdout=os.path.join(outdir,image+'_SN_dao.txt'))


apmag = np.genfromtxt(os.path.join(outdir,image+'_SN_ap.txt'))

SNap = apmag[0]
errSNap = apmag[1]

daomag = np.genfromtxt(os.path.join(outdir,image+'_SN_dao.txt'))

try:
    SNdao = daomag[0]
    errSNdao = daomag[1]
except:
    print 'PSF could not be fit, using aperture mag'
    SNdao = np.nan
    errSNdao = np.nan


print '\n'

if filtername1 in seqMags:

    calMagsDao = SNdao + ZP

    errMagDao = np.sqrt(errSNdao**2 + errZP**2)


    calMagsAp = SNap + ZP

    errMagAp = np.sqrt(errSNap**2 + errZP**2)

else:
    calMagsDao = SNdao

    errMagDao = errSNdao


    calMagsAp = SNap

    errMagAp = errSNap

    comment1 = 'No ZP - instrumental mag only'

    print '> No ZP - instrumental mag only!!!'



print '> PSF mag = '+'%.2f +/- %.2f' %(calMagsDao,errMagDao)
print '> Aperture mag = '+'%.2f +/- %.2f' %(calMagsAp,errMagAp)


comment = raw_input('\n> Add comment to output file: ')

outFile.write('\n'+image+'\t'+filtername1+'\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t'
                %(mjd,calMagsDao,errMagDao,calMagsAp,errMagAp))
outFile.write(comment)


# con = raw_input('\n> Finished... ')

# plt.close(fig1)
# plt.close(fig2)
# plt.close(fig3)
# plt.close(fig4)

outFile.close()

ZPfile.close()


os.remove('im_seq_phot1.txt')
os.remove('tmpl_seq_phot1.txt')


for i in glob.glob('*.mag.*'):
    shutil.copy(i,outdir)

for i in glob.glob('*.psf.*'):
    shutil.copy(i,outdir)

for i in glob.glob('*.mag.*'):
    os.remove(i)

for i in glob.glob('*.psf.*'):
    os.remove(i)

for i in glob.glob('*.sub.*'):
    os.remove(i)

for i in glob.glob('*.als.*'):
    os.remove(i)

for i in glob.glob('*.arj.*'):
    os.remove(i)

for i in glob.glob('*.pst.*'):
    os.remove(i)

for i in glob.glob('*.psg.*'):
    os.remove(i)

for i in glob.glob('*pix*fits'):
    os.remove(i)

for i in glob.glob('*_psf_stars.txt'):
    os.remove(i)

for i in glob.glob('coords'):
    os.remove(i)

for i in glob.glob('template0.fits'):
    os.remove(i)

if not keep:
    for i in glob.glob('*_ped.fits'):
        os.remove(i)
    for i in glob.glob('*_aligned_tmpl.fits'):
        os.remove(i)
    for i in glob.glob('*_ped_conv.fits'):
        os.remove(i)
    for i in glob.glob('geomap*.txt'):
        os.remove(i)
