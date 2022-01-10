#!/usr/bin/env python

'''
    Get sequence star data from PS1, for use with psf.py photometry package
    Requires requests package
'''

import numpy as np
import requests
import glob
import sys
import os

# Note: increased search radius and max_records for UKIRT WFCAM but may want to decrease if it gets too slow

def PS1catalog(ra,dec,magmin,magmax):

    queryurl = 'https://archive.stsci.edu/panstarrs/search.php?'
    queryurl += 'RA='+str(ra)
    queryurl += '&DEC='+str(dec)
    queryurl += '&SR=0.12&selectedColumnsCsv=ndetections,raMean,decMean,'
    queryurl += 'gMeanPSFMag,rMeanPSFMag,iMeanPSFMag,zMeanPSFMag,yMeanPSFMag,iMeanKronMag'
    queryurl += '&ordercolumn1=ndetections&descending1=on&max_records=300'

    print '\nQuerying PS1 for reference stars via MAST...\n'

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

        print 'Success! Sequence star file created: PS1_seq.txt'

    else:
        sys.exit('Field not in PS1! Exiting')



def PS1cutouts(ra,dec,filt,size=2500):

    print '\nSearching for PS1 images of field...\n'

    ps1_url = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?'

    ps1_url += '&ra='+str(ra)
    ps1_url += '&dec='+str(dec)
    ps1_url += '&filters='+filt

    ps1_im = requests.get(ps1_url)

    try:
        image_name = ps1_im.text.split()[17]

        print 'Image found: ' + image_name + '\n'

        cutout_url = 'http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?&filetypes=stack'

        cutout_url += '&size='+str(size)
        cutout_url += '&ra='+str(ra)
        cutout_url += '&dec='+str(dec)
        cutout_url += '&filters='+filt
        cutout_url += '&format=fits'
        cutout_url += '&red='+image_name

        dest_file = filt + '_template.fits'

        cmd = 'wget -O %s "%s"' % (dest_file, cutout_url)

        os.system(cmd)

        print 'Template downloaded as ' + dest_file + '\n'

    except:
        print '\nPS1 template search failed!\n'
