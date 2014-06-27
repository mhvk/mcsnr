import warnings
import numpy as np
from scipy import optimize
from collections import OrderedDict

from astropy import units as u, constants as const
from astropy.table import Table

from geminiutil.gmos.alchemy.mos import MOSSpectrum

from specgrid.composite import ModelStar
from specgrid.plugins import (Interpolate, Normalize, Convolve,
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


def fit_spectrum(self, kwargs):
    """

    :param kwargs:
    :return:
    """

def get_model_spectrum(self, **kwargs):
    
    if self.model_star is None:
        raise AttributeError('No model star associated')
    npol = kwargs.pop('npol', 5)
    
    model = self.model_star.eval(**kwargs)
    return self._normalize(model.wavelength, model.flux, npol=npol)
    


#def get_model_spectrum(self, **kwargs):
#    return self.model_star(**kwargs)

def get_spectral_fit(self, **kwargs):
    npol = kwargs.pop('npol', 5)
    fitter = kwargs.pop('fitter', 'leastsq')
    guess = OrderedDict()
    print self.slice.science_set.science.instrument_setup.grating.name
    print "Fitting with {0} and guess {1}".format(fitter, kwargs)
    for kwarg in kwargs:
        if kwarg not in self.model_star.parameters:
            raise ValueError("Parameter {0} not known in model_star"
                             .format(kwarg))
        guess[kwarg] = kwargs[kwarg]

    def spectral_model_fit(pars):
        pardict = OrderedDict()
        for key, par in zip(guess.keys(), pars):
            pardict[key] = par
        
        model = self.get_model_spectrum(npol=npol, **pardict)
        if np.isnan(model[0]):
            return np.inf
        else:
            if fitter == 'leastsq':
                return ((self.flux - model) / self.uncertainty).to(1).value
            else:
                return np.sum(((self.flux - model) / self.uncertainty)**2).to(1).value

    if fitter == 'leastsq':
        fit = optimize.leastsq(spectral_model_fit, np.array(guess.values()),
                            full_output=True)

        stellar_params = OrderedDict((key, par) for key, par in zip(guess.keys(), fit[0]))
        if fit[1] is not None:
            stellar_params_uncertainty = OrderedDict(
                (key, np.sqrt(par)) for key, par in
                 zip(guess.keys(), np.diag(fit[1])))
        else:
            stellar_params_uncertainty = OrderedDict((key, None) for key in guess.keys())
    else:
        fit =  optimize.minimize(spectral_model_fit, np.array(guess.values()), method=fitter)
        stellar_params = OrderedDict((key, par) for key, par in zip(guess.keys(), fit['x']))
        stellar_params_uncertainty = OrderedDict((key, None) for key, par in zip(guess.keys(), fit['x']))

    return stellar_params, stellar_params_uncertainty, fit

class SimpleStellarParametersFit(object):

    def __init__(self, model, spectrum, npol=5):
        self.model = model
        self.spectrum = spectrum
        self.spectrum.signal_to_noise = (self.spectrum.flux /
                                         self.spectrum.uncertainty)

        self.spectrum._Vp = np.polynomial.polynomial.polyvander(
            self.spectrum.wavelength/self.spectrum.wavelength.mean() - 1.,
            npol)

    def __call__(self, pars):
        par_dict = {}
        for key, value in zip(self.model.parameters, pars):
            par_dict[key] = value

        model_spec = self.model.eval(**par_dict)

        normalized_model = self.spectrum._normalize(model_spec.wavelength,
                                                    model_spec.flux)
        


def _spectral_fit(self, model_wavelength, model_flux, velocity):
    # Doppler shift the model grid
    velocity = u.Quantity(velocity, u.Unit('km/s'))
    shifted_wavelength = (model_wavelength.to(self.wavelength.unit) *
                          (1 + velocity/const.c))

    # interpolate Doppler-shifted model on the observed wavelengths
    interpolated_model = np.interp(self.wavelength.value,
                                   shifted_wavelength.value,
                                   model_flux)

    rcond = len(self.flux)*np.finfo(self.flux.dtype).eps
    # V[:,0]=mfi/e, Vp[:,1]=mfi/e*w, .., Vp[:,npol]=mfi/e*w**npol
    V = self._Vp * (interpolated_model/self.uncertainty)[:,np.newaxis]
    # normalizes different powers
    scl = np.sqrt((V*V).sum(0))
    sol, resids, rank, s = np.linalg.lstsq(V/scl, self.signal_to_noise,
                                           rcond)
    sol = (sol.T/scl).T
    if rank != self._Vp.shape[-1] - 1:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg)

    fit = np.dot(V, sol) * self.uncertainty
    chi2 = np.sum(((self.flux-fit)/self.uncertainty)**2)
    return fit, chi2, interpolated_model


def _normalize(self, model_wavelength, model_flux, npol=5):
    if getattr(self, 'signal_to_noise', None) is None:
        self.signal_to_noise = (self.flux / self.uncertainty)
    
    if getattr(self, '_Vp', None) is None or self._Vp.shape[-1] != npol:
        self._Vp = np.polynomial.polynomial.polyvander(
            self.wavelength/self.wavelength.mean() - 1., npol)
    
    # interpolate Doppler-shifted model on the observed wavelengths
    interpolated_model = np.interp(self.wavelength,
                                   model_wavelength.value,
                                   model_flux)

    rcond = len(self.flux)*np.finfo(self.flux.dtype).eps
    # V[:,0]=mfi/e, Vp[:,1]=mfi/e*w, .., Vp[:,npol]=mfi/e*w**npol
    V = self._Vp * (interpolated_model/self.uncertainty)[:,np.newaxis]
    # normalizes different powers
    scl = np.sqrt((V*V).sum(0))
    sol, resids, rank, s = np.linalg.lstsq(V/scl, self.signal_to_noise,
                                           rcond)
    sol = (sol.T/scl).T
    if rank != self._Vp.shape[-1] - 1:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg)

    fit = np.dot(V, sol) * self.uncertainty
    # chi2 = np.sum(((self.flux-fit)/self.uncertainty)**2)
    return fit.value * self.flux.unit


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
    norm = Normalize(self.to_spectrum_1d(), npol)

    model_star = ModelStar([spectral_grid, rot, doppler, interp, norm])

    return model_star

def to_spectrum_1d(self):
    return Spectrum1D.from_array(self.wavelength.value, self.flux.value,
                          unit=self.flux.unit,
                          dispersion_unit=self.wavelength.unit)

MOSSpectrum.init_spectral_parameters = init_spectral_parameters
MOSSpectrum.to_spectrum_1d = to_spectrum_1d
MOSSpectrum.get_model_star = get_model_star
MOSSpectrum.get_spectral_fit = get_spectral_fit
MOSSpectrum._spectral_fit = _spectral_fit
MOSSpectrum._normalize = _normalize
MOSSpectrum.get_model_spectrum = get_model_spectrum
MOSSpectrum.find_velocity = find_velocity
