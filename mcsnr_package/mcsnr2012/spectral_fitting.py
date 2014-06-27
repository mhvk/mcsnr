# import warnings
import numpy as np
from scipy import optimize
from collections import OrderedDict

from astropy import units as u
from astropy.table import Table

from geminiutil.gmos.alchemy.mos import MOSSpectrum

from specgrid.composite import ModelStar
from specgrid.plugins import (Interpolate, NormalizeParts,
                              RotationalBroadening, DopplerShift)

from specutils import Spectrum1D

parameters = {'teff', 'logg', 'feh', 'vrot', 'vrad'}


def init_spectral_parameters(self, **kwargs):
    """
    Initializing parameters for spectrum

    Parameters
    ----------

    kwargs :

    """

    self.spectral_parameters = OrderedDict([('teff', 5780.), ('logg', 4.4),
                                            ('feh', 0.0), ('vrot', 1e-5),
                                            ('vrad', 0)])

    self.spectral_parameters.update(OrderedDict(
        [(key + '_uncertainty', np.nan) for key in self.spectral_parameters]))

    self.spectral_parameters.update(OrderedDict(
        [(key + '_fixed', False) for key in self.spectral_parameters]))

    self.spectral_parameters['npol'] = 5
    self.spectral_parameters.update(kwargs)


def get_synthetic_spectrum(self):
    for param in parameters:
        setattr(self.model_star, param, self.spectral_parameters[param])

    return self.model_star()


def fit_spectrum(self, kwargs):
    """

    :param kwargs:
    :return:
    """


def find_velocity(self, teff, logg, feh,
                  velocities=np.linspace(-400, 400, 51) * u.km / u.s,
                  npol=3, sigrange=None, vrange=70):

    teff = min(max(teff, 4000), 35000)

    model_wavelength, model_flux = self.get_model_spectrum(teff, logg, feh)
    # pre-calculate observed flux/error and polynomial bases
    if not hasattr(self, 'signal_to_noise'):
        self.signal_to_noise = (self.flux / self.uncertainty)
    # Vp[:,0]=1, Vp[:,1]=w, .., Vp[:,npol]=w**npol
    if not hasattr(self, '_Vp') or self._Vp.shape[-1] != npol + 1:
        self._Vp = np.polynomial.polynomial.polyvander(
            self.wavelength/self.wavelength.mean() - 1., npol)

    chi2 = Table([velocities, np.zeros_like(velocities.value)],
                 names=['velocity','chi2'])

    chi2['chi2'] = np.array([
        self._spectral_fit(model_wavelength, model_flux, v)[1]
        for v in velocities])
    chi2.meta['ndata'] = len(self.flux)
    chi2.meta['npar'] = npol+1+1
    chi2.meta['ndof'] = chi2.meta['ndata']-chi2.meta['npar']

    if vrange is None and sigrange is None or len(velocities) < 3:
        ibest = chi2['chi2'].argmin()
        vbest, bestchi2 = chi2[ibest]
        chi2.meta['vbest'] = vbest
        chi2.meta['verr'] = 0.
        chi2.meta['bestchi2'] = bestchi2
    else:
        vbest, verr, bestchi2 = minchi2(chi2, vrange, sigrange)

    fit, bestchi2, interpolated_model = self._spectral_fit(model_wavelength,
                                                           model_flux,
                                                           vbest)

    return vbest, verr, fit, chi2, interpolated_model


def minchi2(chi2, vrange=None, sigrange=None, fitcol='chi2fit'):
    assert vrange is not None or sigrange is not None
    if sigrange is None:
        sigrange = 1e10
    if vrange is None:
        vrange = 1e10

    iminchi2 = chi2['chi2'].argmin()
    ndof = float(chi2.meta['ndof'])
    iok = np.where((chi2['chi2'] <
                    chi2['chi2'][iminchi2]*(1.+sigrange**2/ndof)) &
                   (abs(chi2['velocity']-chi2['velocity'][iminchi2]) <= vrange))

    p = np.polynomial.Polynomial.fit(chi2['velocity'][iok], chi2['chi2'][iok],
                                     2, domain=[])

    vbest = -p.coef[1]/2./p.coef[2]
    # normally, get sigma from where chi2 = chi2min+1, but best to scale
    # errors, so look for chi2 = chi2min*(1+1/ndof) ->
    # a verr**2 = chi2min/ndof -> verr = sqrt(chi2min/ndof/a)
    verr = np.sqrt(p(vbest)/p.coef[2]/ndof)
    chi2.meta['vbest'] = vbest
    chi2.meta['verr'] = verr
    chi2.meta['bestchi2'] = p(vbest)
    if fitcol is not None:
        chi2[fitcol] = p(chi2['velocity'])

    return chi2.meta['vbest'], chi2.meta['verr'], chi2.meta['bestchi2']


def get_model_star(self, spectral_grid, npol=5):
    """
    Initialize model_star

    Parameters
    ----------

    spectral_grid : BaseSpectralGrid


    """

    rot = RotationalBroadening()
    doppler = DopplerShift()
    interp = Interpolate(self.to_spectrum_1d())
    assert len(self.wavelength) == self.table['wavelength'].size()
    one_chip_size = len(self.table['x'])
    parts = [slice(None, one_chip_size),
             slice(one_chip_size, 2*one_chip_size),
             slice(2*one_chip_size, None)]
    norm = NormalizeParts(self.to_spectrum_1d(), parts=[parts], npol=3)

    model_star = ModelStar([spectral_grid, rot, doppler, interp, norm])

    return model_star


def to_spectrum_1d(self):
    return Spectrum1D.from_array(self.wavelength.value, self.flux.value,
                                 unit=self.flux.unit,
                                 dispersion_unit=self.wavelength.unit)

MOSSpectrum.init_spectral_parameters = init_spectral_parameters
MOSSpectrum.to_spectrum_1d = to_spectrum_1d
MOSSpectrum.get_model_star = get_model_star
MOSSpectrum.find_velocity = find_velocity
