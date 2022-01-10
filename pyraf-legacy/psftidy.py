import glob
import os
import shutil

for i in glob.glob('*.mag.*'):
    shutil.copy(i,outdir)

for i in glob.glob('*.psf.*'):
    shutil.copy(i,outdir)

for i in glob.glob('*.mag.*'):
    os.remove(i)

for i in glob.glob('*.psf.*'):
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

for i in glob.glob('*.sub.*'):
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

for i in glob.glob('coords'):
    os.remove(i)

for i in glob.glob('geomap*.txt'):
    os.remove(i)

for i in glob.glob('template0.fits'):
    os.remove(i)

for i in glob.glob('*_ped.fits'):
    os.remove(i)
for i in glob.glob('*_aligned_tmpl.fits'):
    os.remove(i)
for i in glob.glob('*_ped_conv.fits'):
    os.remove(i)
