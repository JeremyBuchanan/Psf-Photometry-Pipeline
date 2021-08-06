This is a pipeline for reducing raw imaging data using point spread function
<<<<<<< HEAD
photometry. This pipeline has until now been used to reduce multi-color
photometry data from the Las Cumbres Observatory Global Telescope (LCOGT).
=======
photometry. This pipeline has until now been used to reduce data from the Los
Cumbras Observatory Global Telescope (LCOGT).
>>>>>>> 48510395289ba6f5538316af7f3d7c4ab5c912ff

# Setup

Drop this repository into a directory which includes a "results" folder, as well
as a "fits" folder, which includes all the FITS files of the data.
Additionally, a CSV file is needed in the same directory, which includes
columns of the names of the FITS files, as well as an id that groups the files
into sets of images, if needed.

## obj_data

This file needs to be updated with the following information regarding the
target object:

- ra
- dec
- pmra
- pmdec
- plx
- epoch

# Changelog

## Version 0.1.1
Some bug fixes:
- find_fwhm function now handles timeouts in curve_fit
- get_wcs function now handles multiple exceptions, including timeouts and
  failures in solving the WCS transform
- pipeline no longer tries to finish if median_image = None
- added p_io file for pipeline outputs, and simplified some things in the outputs
