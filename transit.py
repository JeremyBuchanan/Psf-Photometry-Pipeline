import copy as copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psfpkg as p
import time as t
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import ZScaleInterval, SqrtStretch, ImageNormalize
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams.update({'font.size': 12})

filenames, epochid = np.loadtxt('./transit.csv', unpack=True, delimiter=',', usecols=(0,1), dtype='U100,f')
nepochs = np.int(np.max(epochid))
plot_residuals = input('Plot residuals? [y/n] ')
for i in range(0, nepochs+1):
    images = filenames[(epochid == i)]
    x = str(images[0])
    t0 = t.perf_counter()
    set_name = x[:22]
    path = './fits_t/'
    im_data, headers = p.import_images(im_list=images, p=path)
    im_data[0] = im_data[0][0:2500,:]
    im_copy = copy.deepcopy(im_data[0])
    fwhm, im_sig = p.find_fwhm(image=im_copy, size=50)
    if im_sig == 0:
        image = im_data[0]
        print('No stars were detected in this image set.')
        pp = PdfPages('./results_t/'+set_name+'_'+'nostars.pdf')
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
        mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=2.0)
        image -= median_val
        norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=SqrtStretch())
        stars_tbl = Table()
        t1 = t.perf_counter()
        sources = p.find_stars(image=image, sigma=im_sig)
        t2 = t.perf_counter()
        stars_tbl = p.image_mask(image=image, sources=sources, fwhm=fwhm, bkg=median_val, bkg_std=std_val)
        image_lbs = p.bkg_sub(image=image, stars_tbl=stars_tbl, fwhm=fwhm)
        epsf, stars, fitted_stars = p.build_psf(image=image_lbs, stars_tbl=stars_tbl, fwhm=fwhm)
        t3 = t.perf_counter()
        if len(stars_tbl) <= 10 or fwhm > 30:
            print('FAILED!')
            results = []
            p.write_pdf_f(name='./results_t/'+set_name+'_'+'.pdf', images=im_data, stars=stars,
                            model=epsf.data, plot_res=plot_residuals)
            p.write_txt_f(name='./results_t/'+set_name+'_'+'_diag.txt', sources=sources,
                            stars_tbl=stars_tbl, fwhm=fwhm, results=results)
        else:
            results, photometry = p.do_photometry(image=image, epsf=epsf, fwhm=fwhm)
            results_tbl, residual_stars, final_stars = p.get_residuals(results=results, photometry=photometry,
                                                                    fwhm=fwhm, image=image)
            results.sort('flux_fit', reverse=True)
            t4 = t.perf_counter()
            sky, wcs_header = p.get_wcs(results_tbl=results_tbl)
            t5 = t.perf_counter()
            avg_airmass, bjd, header = p.write_fits(fn='./results_t/'+set_name+'_'+'.fits',
                                                    data=image, im_headers=headers, wcs_header=wcs_header)
            p.write_pdf(name='./results_t/'+set_name+'_'+'.pdf', images=im_data, model=epsf.data,
                        final_stars=final_stars, residual_stars=residual_stars, fluxes=results_tbl['flux'],
                        plot_res=plot_residuals)
            p.write_txt(name='./results_t/'+set_name+'_'+'_diag.txt', sources=sources, stars_tbl=stars_tbl,
                        results=results, fwhm=fwhm,t0=t0,t1=t1,t2=t2,t3=t3,t4=t4,t5=t5)
            p.write_csv(name='./results_t/'+set_name+'_'+'.csv', im_name=set_name,
                        bjd=bjd[0], filt=header['FILTER'], airmass=avg_airmass, results=results, sky=sky)
