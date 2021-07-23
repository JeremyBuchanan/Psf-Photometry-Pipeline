import copy as copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import core as c
import time as t
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import ZScaleInterval, SqrtStretch, ImageNormalize
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams.update({'font.size': 12})

def pipeline(fn_path, fits_path, res_path):
    '''
    This is a pipeline for reducing raw imaging data using point spread function
    photometry

        Parameters
        ----------
        fn_path: string
            pathway to the csv containing the names of all image files in one
            column, and the grouping id of each image in the second column
        fits_path: string
            pathway to a folder containing all the FITS files of the imaging
            im_data
        res_path: string
            pathway to a directory for all the output files of the photometry
            routine

        Returns
        -------
        N/A

        Outputs
        -------
        CSV: tabulated data of the photometry results
        FITS: contains the header as well as the imaging data for the median
        combined images of each image set
        PDF: images of all the images in the image set, as well as the
        constructed PSF and (per input) the residual images of all the stars
        included in the calculations
        TXT: text file containing a bunch of diagnostic information about the
        script
    '''
    filenames, epochid = np.loadtxt(fn_path, unpack=True, delimiter=',', usecols=(0,1), dtype='U100,f')
    nepochs = np.int(np.max(epochid))
    plot_residuals = input('Plot residuals? [y/n] ')

    for i in range(0, nepochs+1):

        images = filenames[(epochid == i)]
        x = str(images[0])
        t0 = t.perf_counter()
        set_name = x[:22]
        path = fits_path
        im_data, headers = c.import_images(im_list=images, p=path)
        im_copy = copy.deepcopy(im_data[0])
        fwhm, im_sig = c.find_fwhm(image=im_copy)

        if im_sig == 0:
            print('No stars were detected in this image set.')
            pp = PdfPages(res_path+set_name+'_'+np.str(i)+'nostars.pdf')
            for i in range(len(im_data)):
                fig, ax = plt.subplots(1, figsize=(10, 10))
                norm = ImageNormalize(im_data[i], interval=ZScaleInterval(), stretch=SqrtStretch())
                im = ax.imshow(im_data[i], norm=norm)
                plt.colorbar(im)
                plt.tight_layout()
                pp.savefig()
                plt.close()
            pp.close()

        else:
            if len(im_data) > 1:
                median_image = c.image_combiner(im_data=im_data, im_sig=im_sig)

                if median_image is None:
                    image = im_data[0]
                    print('No stars were detected in this image set.')
                    pp = PdfPages(res_path+set_name+'_'+np.str(i)+'nostars.pdf')
                    for i in range(len(im_data)):
                        fig, ax = plt.subplots(1, figsize=(10, 10))
                        norm = ImageNormalize(im_data[i], interval=ZScaleInterval(), stretch=SqrtStretch())
                        im = ax.imshow(im_data[i], norm=norm)
                        plt.colorbar(im)
                        plt.tight_layout()
                        pp.savefig()
                        plt.close()
                    pp.close()
                    continue

                else:
                    image = median_image

            mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=2.0)
            image -= median_val
            norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=SqrtStretch())
            stars_tbl = Table()
            t1 = t.perf_counter()
            sources = c.find_stars(image=image, sigma=im_sig)
            t2 = t.perf_counter()
            stars_tbl = c.image_mask(image=image, sources=sources, fwhm=fwhm, bkg=median_val, bkg_std=std_val)
            image_lbs = c.bkg_sub(image=image, stars_tbl=stars_tbl, fwhm=fwhm)
            epsf, stars, fitted_stars = c.build_psf(image=image_lbs, stars_tbl=stars_tbl, fwhm=fwhm)
            t3 = t.perf_counter()

            if len(stars_tbl) <= 10 or fwhm > 30:
                print('Not enough stars were detected.')
                results = []
                c.write_pdf_f(name=res_path+set_name+'_'+np.str(i)+'.pdf', images=im_data, stars=stars,
                                model=epsf.data, plot_res=plot_residuals)
                c.write_txt_f(name=res_path+set_name+'_'+np.str(i)+'_diag.txt', sources=sources,
                                stars_tbl=stars_tbl, fwhm=fwhm, results=results)

            else:
                results, photometry = c.do_photometry(image=image, epsf=epsf, fwhm=fwhm)
                results_tbl, residual_stars, final_stars = c.get_residuals(results=results, photometry=photometry, fwhm=fwhm, image=image)
                results.sort('flux_fit', reverse=True)
                t4 = t.perf_counter()
                sky, wcs_header = c.get_wcs(results_tbl=results_tbl)
                t5 = t.perf_counter()
                avg_airmass, bjd, header = c.write_fits(fn=res_path+set_name+'_'+np.str(i)+'.fits', data=image, im_headers=headers, wcs_header=wcs_header)

                c.write_pdf(name=res_path+set_name+'_'+np.str(i)+'.pdf', images=im_data, model=epsf.data, final_stars=final_stars, residual_stars=residual_stars, fluxes=results_tbl['flux'], plot_res=plot_residuals)
                c.write_txt(name=res_path+set_name+'_'+np.str(i)+'_diag.txt', sources=sources, stars_tbl=stars_tbl, results=results, fwhm=fwhm,t0=t0,t1=t1,t2=t2,t3=t3,t4=t4,t5=t5)
                c.write_csv(name=res_path+set_name+'_'+np.str(i)+'.csv', im_name=set_name, bjd=bjd[0], filt=header['FILTER'], airmass=avg_airmass, results=results, sky=sky)
