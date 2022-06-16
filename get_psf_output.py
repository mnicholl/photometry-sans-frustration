#!/usr/bin/env python

import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--save','-s', dest='save', default=False, action='store_true',
                    help='save light curve to single json')

args = parser.parse_args()

save = args.save

filter_order = {'u': 4, 'g': 6, 'r': 8,
                'i': 10, 'z': 12, 'y':13, 'U': 3,
                'B': 5, 'V': 7, 'R': 9,
                'I': 11, 'J': 14, 'H': 15,
                'K': 16, 'UVW2': 0, 'UVM2': 1,
                'UVW1': 2}


cols = {'u': 'dodgerblue', 'g': 'g', 'r': 'r',
        'i': 'goldenrod', 'z': 'k', 'y':'hotpink', 'U': 'midnightblue',
        'B': 'b', 'V': 'yellowgreen', 'R': 'crimson',
        'I': 'chocolate', 'J': 'darkred', 'H': 'orangered',
        'K': 'saddlebrown', 'UVW2': 'mediumorchid', 'UVM2': 'purple',
        'UVW1': 'cadetblue'}

phot = {}

plt.figure(1)

plt.clf()

datfiles = []

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if 'PSF_phot' in name:
            datfiles.append(os.path.join(root, name))



for i in datfiles:
    f = open(i)
    for j in f.readlines():
        if j[0]!='#':
            data = j.split('\t')
            band = data[1]
            if band not in phot:
                phot[band] = {'PSFerr': [], 'PSFmag': [], 'Aperr': [], 'Apmag': [], 'ApOpterr': [], 'ApOptmag': [], 'mjd': []}
            phot[band]['mjd'].append(float(data[2]))
            phot[band]['PSFmag'].append(float(data[3]))
            phot[band]['PSFerr'].append(float(data[4]))
            phot[band]['ApOptmag'].append(float(data[5]))
            phot[band]['ApOpterr'].append(float(data[6]))
            phot[band]['ApBigmag'].append(float(data[7]))
            phot[band]['ApBigerr'].append(float(data[8]))


for band in phot:
    if band in cols:
        plt.errorbar(phot[band]['mjd'],phot[band]['PSFmag']
        ,phot[band]['PSFerr'],fmt='o',markersize=10,mfc=cols[band],
        markeredgecolor=cols[band],ecolor=cols[band],label=band)
    else:
        plt.errorbar(phot[band]['mjd'],phot[band]['PSFmag']
        ,phot[band]['PSFerr'],fmt='o',markersize=10,mfc='k',
        markeredgecolor='k',label=band)

handles, labels = plt.gca().get_legend_handles_labels()

order = []
for i in labels: order.append(filter_order[i])

labels = [i for j,i in sorted(zip(order,labels))]
handles = [i for j,i in sorted(zip(order,handles))]

plt.legend(handles,labels,ncol=2,columnspacing=0.2)

plt.xlabel('MJD')

plt.ylabel('Apparent magnitude')

plt.gca().invert_yaxis()

plt.tight_layout(pad=0.5)

plt.show()

if save:
    json.dump(phot, open('PSF_lightcurve_'+str(len(glob.glob('PSF_lightcurve*.txt')))+'.json','w'))
    
    outfile = 'PSF_lightcurve_'
    
    for i in labels:
        outfile += i
        
    outfile += '.txt'
    
    f = open(outfile,'w')
    
    f.write('#MJD\tPSF\terr\tOptAp\terr\tBigAp\terr\tband\n')

    for i in labels:
        for j in range(len(phot[i]['mjd'])):
            f.write('%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n' %(phot[i]['mjd'][j], phot[i]['PSFmag'][j], phot[i]['PSFerr'][j], phot[i]['ApOptmag'][j], phot[i]['ApOpterr'][j], phot[i]['ApBigmag'][j], phot[i]['ApBigerr'][j], i))

    f.close()
