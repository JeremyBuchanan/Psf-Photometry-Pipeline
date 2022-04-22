This is a pipeline for reducing raw imaging data using point spread function photometry. This pipeline has until now been used to reduce multi-color photometry data from the Las Cumbres Observatory Global Telescope (LCOGT).

# Setup

Clone this repository into a directory which includes a "results" folder, as well as a "fits" folder, which includes all the FITS files of the data. 

Additionally, a separate CSV file containing metadata is needed in the same directory. I set up the pipeline to read in two columns. The first column simply contains the names of all the FITS files. My dataset had images that were grouped into sets of three, and each set needed to be imported together. Therefore, the second column contains an integer id that indicates which image set each file is a part of.

## obj_data

This file needs to be updated with the following information regarding the target object:

- ra (right ascension)
- dec (declination)
- pmra (proper motion in right ascension)
- pmdec (proper motion in declination)
- plx (parallax)
- epoch

# Changelog

## Version 0.1.1
Some bug fixes:
- find_fwhm function now handles timeouts in curve_fit
- get_wcs function now handles multiple exceptions, including timeouts and
  failures in solving the WCS transform
- pipeline no longer tries to finish if median_image = None
- added p_io file for pipeline outputs, and simplified some things in the outputs

## Version 0.1.2
When I originally packaged this code, I dealt with a ton of bugs that arose in the environment by just rolling back to previous versions of the packages I was using instead of actually fixing stuff. Here I actually tried to fix stuff, all the mean while changing from python 3.8 to 3.9.
- Added a notebook with a walkthrough documenting my process for turning raw data into actual results
- Created a new requirements.txt file, so I can setup the environment with the python venv module instead of needing conda and creating a conda environment from the old yml file
- Deleted the old yml file
- Attempted to fix an error from the residuals pdf being too large
- Fixed an unbound variable error for sets of image sets only containing one image