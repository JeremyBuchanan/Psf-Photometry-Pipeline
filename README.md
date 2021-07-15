This is a pipeline for reducing raw imaging data using point spread function
photometry. This pipeline has until now been used to reduce data from the Los
Cumbras Global Observatory (LCOGT).

# Setup

Drop this repository into a directory which includes a "results" file, as well
as a "fits" file, which includes all the FITS files of the data.
Additionally, a CSV file is needed in the same directory, which includes
columns of the names of the FITS files, as well as an id that groups the files
into sets of images, if needed.

## obj_data

This file needs to be updated with certain information regarding the
target object. Pertinent info includes:

- ra: right ascension
- dec: declination
- pmra: proper motion right ascension
- pmdec: proper motion declination
- plx: paralax
- epoch
