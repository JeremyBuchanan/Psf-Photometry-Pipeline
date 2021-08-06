import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import obj_data as od
import saphires as saph
from astropy.time import Time
from astropy.visualization import ZScaleInterval, SqrtStretch, ImageNormalize
from matplotlib.backends.backend_pdf import PdfPages

ra = od.ra
dec = od.dec
pmra = od.pmra
pmdec = od.pmdec
plx  = od.plx
epoch = od.epoch
matplotlib.rcParams.update({'font.size': 12})

def write_fits(fn, data, im_headers, wcs_header):
    '''
    Writes a new fits file including the image data and
    and updated header for the new image

        Parameters
        ----------
        fn: string
            The desired file name of the new fits file
        data: array-like
            Contains all the image data

        Returns
        -------
        avg_airmass: float
            the amount of atmosphere obscuring the target, found in image header. Here
            the airmass for all images is averaged
        bjd: float
            Barycentric Julian Date, found in the image header
        header: Header
    '''
    for keys in wcs_header:
        if keys not in ['HISTORY', 'COMMENT']:
            im_headers[0][keys] = wcs_header[keys]
    airmass = []
    for i in im_headers:
        airmass.append(i['AIRMASS'])
    avg_airmass = np.mean(airmass)
    im_headers[0]['AIRMASS'] = avg_airmass
    jd_middle = np.zeros(len(im_headers))
    for i in range(len(im_headers)):
        jd_middle[i] = Time(im_headers[i]['DATE-OBS'], format='isot').jd
        exptime = im_headers[i]['EXPTIME']
        jd_middle[i] = jd_middle[i] + (exptime/2.0)/3600.0/24.0
    isot_date_obs = Time(np.mean(jd_middle), format='jd').isot
    tele = im_headers[0]['SITEID']
    brv,bjd,bvcorr = saph.utils.brvc(isot_date_obs,0.0,tele,ra=ra,dec=dec,epoch=epoch, pmra=pmra, pmdec=pmdec, px=plx)
    im_headers[0]['BJD'] = bjd[0]
    header = im_headers[0]
    hdu_p = fits.PrimaryHDU(data=data, header=header)
    hdu = fits.HDUList([hdu_p])
    hdu.writeto(fn)

    return avg_airmass, bjd, header

def write_pdf(name, images, model=None, final_stars=None, residual_stars=None, fluxes=None, plot_res=None):
    pp = PdfPages(name)
    for i in range(len(images)):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        norm = ImageNormalize(images[i], interval=ZScaleInterval(), stretch=SqrtStretch())
        im = ax.imshow(images[i], norm=norm)
        plt.colorbar(im)
        plt.tight_layout()
        pp.savefig()
        plt.close()
    if model is not None:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        psf = ax.imshow(model)
        plt.colorbar(psf)
        ax.set_title('PSF Model')
        plt.tight_layout()
        pp.savefig()
        plt.close()
    if final_stars is not None:
        if plot_res == 'y':
            nrows = len(final_stars)
            ncols = 2
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 800), squeeze=True)
            ax = ax.ravel()
            index = 0
            for i in range(0, nrows*ncols, 2):
                norm = simple_norm(final_stars[index],'log')
                norm2 = simple_norm(residual_stars[index], 'linear')
                im = ax[i].imshow(final_stars[index], norm=norm, origin='lower', cmap='viridis', interpolation='none')
                fig.colorbar(im, ax = ax[i])
                ax[i].set_title(np.str(fluxes[index]))
                im_r = ax[i+1].imshow(residual_stars[index], norm=norm2, origin='lower', cmap='viridis', interpolation='none')
                fig.colorbar(im_r, ax = ax[i+1])
                index = index + 1
            plt.tight_layout()
            pp.savefig()
            plt.close()
    pp.close()

def write_csv(name, im_name, bjd, filt, airmass, results, sky):
    f = open(name, 'w')
    f.write('NAME, ID, BJD, FLUX, FLUX ERROR, MAG, MAG ERROR, FILTER, X POSITION, Y POSITION, AIRMASS, RA, DEC\n')
    for i in range(sky.size):
        if results['flux_fit'][i] > 0:
            star_id = results['id'][i]
            flux = results['flux_fit'][i]
            fluxerr = results['flux_unc'][i]
            mag = -2.5*np.log10(flux)
            magerr = (1.08574*fluxerr)/(flux)
            x_pos = results['x_fit'][i]
            y_pos = results['y_fit'][i]
            ra = sky[i].ra.degree
            dec = sky[i].dec.degree
            f.write(im_name+','+np.str(i)+','+np.str(bjd)+','+np.str(flux)+','+np.str(fluxerr)+','+np.str(mag)+','+np.str(magerr)
                    +','+filt+','+np.str(x_pos)+','+np.str(y_pos)+','+str(airmass)+','+np.str(ra)+','+np.str(dec)+'\n')
    f.close()

def write_txt(name, sources, stars_tbl, fwhm, results=None, t0=None,t1=None,t2=None,t3=None,t4=None,t5=None):
    '''
    Short text file with diagnostic info about each image set, specifically
    for a successful run of the image set

        Parameters
        ----------
        name: string
            name of the saved file
        sources: Table
            tabulated info about all the stars found on the image
        stars_tbl: Table
            tabulated info about all the stars used to form a psf
        results: Table
            tabulated info about all the stars found with the photometry routine

    '''
    f = open(name, 'w')
    f.write('Number of stars in sources: '+np.str(len(sources))+'\nNumber of stars in stars_tbl: '+np.str(len(stars_tbl))
            +'\nNumbers of stars in results: '+np.str(len(results))+'\nMin, Max, Median peaks in sources: '
            +np.str(np.min(sources['peak']))+', '+np.str(np.max(sources['peak']))+', '+np.str(np.median(sources['peak']))
            +'\nMin, Max, Median fluxes in results: '+np.str(np.min(results['flux_fit']))+', '+np.str(np.max(results['flux_fit']))+', '
            +np.str(np.median(results['flux_fit']))+'\nFWHM: '+np.str(fwhm)+'\n')
    if t5:
        t_1 = t1-t0
        t_2 = t2-t1
        t_3 = t3-t2
        t_4 = t4-t3
        t_5 = t5-t4
        t_f = t5-t0
        f.write('Time to combine images: '+np.str(t_1)+'\nTime to find stars: '+np.str(t_2)+'\nTime to build psf: '
                +np.str(t_3)+'\nTime to run photometry: '+np.str(t_4)+'\nTime to get wcs: '+np.str(t_5)+'\nTotal time: '
                +np.str(t_f)+'\n')
    f.close()
