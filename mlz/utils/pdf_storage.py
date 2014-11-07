__author__ = 'Matias Carrasco Kind'
"""
.. module:: pdf_storage
.. moduleauthor:: Matias Carrasco Kind
"""
__author__ = 'Matias Carrasco Kind'
from numpy import *
from scipy import linalg as sla
from scipy.optimize import leastsq
from scipy import special
import pyfits as pf


def sparse_basis(dictionary, query_vec, n_basis, tolerance=None):
    """
    Compute sparse representation of a vector given Dictionary  (basis)
    for a given tolerance or number of basis. It uses Cholesky decomposition to speed the process and to
    solve the linear operations adapted from Rubinstein, R., Zibulevsky, M. and Elad, M., Technical Report - CS
    Technion, April 2008

    :param float dictionary: Array with all basis on each column, must has shape (len(vector), total basis) and each column must have euclidean l-2 norm equal to 1
    :param float query_vec: vector of which a sparse representation is desired
    :param int n_basis: number of desired basis
    :param float tolerance: tolerance desired if n_basis is not needed to be fixed, must input a large number for n_basis to assure achieving tolerance

    :return: indices, values (2 arrays one with the position and the second with the coefficients)
    """

    a_n = zeros(dictionary.shape[1])
    machine_eps = finfo(dictionary.dtype).eps
    alpha = dot(dictionary.T, query_vec)
    res = query_vec
    idxs = arange(dictionary.shape[1])  # keeping track of swapping
    L = zeros((n_basis, n_basis), dtype=dictionary.dtype)
    L[0, 0] = 1.

    for n_active in xrange(n_basis):
        lam = argmax(abs(dot(dictionary.T, res)))
        if lam < n_active or alpha[lam] ** 2 < machine_eps:
            n_active -= 1
            break
        if n_active > 0:
            # Updates the Cholesky decomposition of dictionary
            L[n_active, :n_active] = dot(dictionary[:, :n_active].T, dictionary[:, lam])
            sla.solve_triangular(L[:n_active, :n_active], L[n_active, :n_active], lower=True, overwrite_b=True)
            v = linalg.norm(L[n_active, :n_active]) ** 2
            if 1 - v <= machine_eps:
                print "Selected basis are dependent or normed are not unity"
                break
            L[n_active, n_active] = sqrt(1 - v)
        dictionary[:, [n_active, lam]] = dictionary[:, [lam, n_active]]
        alpha[[n_active, lam]] = alpha[[lam, n_active]]
        idxs[[n_active, lam]] = idxs[[lam, n_active]]
        # solves LL'x = query_vec as a composition of two triangular systems
        gamma = sla.cho_solve((L[:n_active + 1, :n_active + 1], True), alpha[:n_active + 1], overwrite_b=False)
        res = query_vec - dot(dictionary[:, :n_active + 1], gamma)
        if tolerance is not None and linalg.norm(res) ** 2 <= tolerance:
            break
    a_n[idxs[:n_active + 1]] = gamma
    del dictionary
    #return a_n
    return idxs[:n_active + 1], gamma


def reconstruct_pdf(index, vals, zfine, mu, Nmu, sigma, Nsigma, cut=1.e-5):
    """
    This function reconstruct the pdf from the indices and values and parameters used to create the dictionary with
    Gaussians only

    :param int index: List of indices in the dictionary for the selected bases
    :param float vals: values or coefficients corresponding to the listed indices
    :param float zfine: redshift values from the original pdf or used during the sparse representation
    :param float mu: [min_mu, max_mu] values used to create the dictionary
    :param int Nmu: Number of mu values used to create the dictionary
    :param float sigma: [min_sigma, mas_sigma] sigma values used to create the dictionary
    :param int Nsigma: Number of sigma values
    :param float cut: cut threshold when creating the dictionary

    :return: the pdf normalized so it sums to one
    """

    zmid = linspace(mu[0], mu[1], Nmu)
    sig = linspace(sigma[0], sigma[1], Nsigma)
    pdf = zeros(len(zfine))
    for k in xrange(len(index)):
        i = index[k] / Nsigma
        j = index[k] % Nsigma
        pdft = 1. * exp(-((zfine - zmid[i]) ** 2) / (2. * sig[j] * sig[j]))
        pdft = where(pdft >= cut, pdft, 0.)
        pdft = pdft / linalg.norm(pdft)
        pdf += pdft * vals[k]
        #pdf = where(pdf >= cut, pdf, 0)
    pdf = where(greater(pdf, max(pdf) * 0.005), pdf, 0.)
    if sum(pdf) > 0: pdf = pdf / sum(pdf)
    return pdf


def reconstruct_pdf_f(index, vals, zfine, mu, Nmu, sigma, Nsigma):
    """
    This function returns the reconstructed pdf in a functional analytical form, to be used in a analytical form"

    :param int index: List of indices in the dictionary for the selected bases
    :param float vals: values or coefficients corresponding to the listed indices
    :param float zfine: redshift values from the original pdf or used during the sparse representation
    :param float mu: [min_mu, max_mu] values used to create the dictionary
    :param int Nmu: Number of mu values used to create the dictionary
    :param float sigma: [min_sigma, mas_sigma] sigma values used to create the dictionary
    :param int Nsigma: Number of sigma values

    :return: a function representing the pdf
    """

    zmid = linspace(mu[0], mu[1], Nmu)
    sig = linspace(sigma[0], sigma[1], Nsigma)
    pdf = zeros(len(zfine))

    def f(x):
        ft = 0.
        for k in xrange(len(index)):
            i = index[k] / Nsigma
            j = index[k] % Nsigma
            pdft = 1. * exp(-((zfine - zmid[i]) ** 2) / (2. * sig[j] * sig[j]))
            ft2 = 1. * exp(-((x - zmid[i]) ** 2) / (2. * sig[j] * sig[j]))
            #pdft = where(pdft >= cut, pdft, 0.)
            pdft = where(greater(pdft, max(pdft) * 0.005), pdft, 0.)
            ft += 1. / linalg.norm(pdft) * ft2 * vals[k]
        return ft

    return f


def create_gaussian_dict(zfine, mu, Nmu, sigma, Nsigma, cut=1.e-5):
    """
    Creates a gaussian dictionary only

    :param float zfine: the x-axis for the PDF, the redshift resolution
    :param float mu: [min_mu, max_mu], range of mean for gaussian
    :param int Nmu: Number of values between min_mu and max_mu
    :param float sigma: [min_sigma, max_sigma], range of variance for gaussian
    :param int Nsigma: Number of values between min_sigma and max_sigma
    :param float cut: Lower cut for gaussians

    :return: Dictionary as numpy array with shape (len(zfine), Nmu*Nsigma)
    :rtype: float
    """

    zmid = linspace(mu[0], mu[1], Nmu)
    sig = linspace(sigma[0], sigma[1], Nsigma)
    NA = Nmu * Nsigma
    Npdf = len(zfine)
    A = zeros((Npdf, Nmu * Nsigma))
    k = 0
    for i in xrange(Nmu):
        for j in xrange(Nsigma):
            pdft = 1. * exp(-((zfine - zmid[i]) ** 2) / (2. * sig[j] * sig[j]))
            pdft = where(pdft >= cut, pdft, 0.)
            #pdft = where(greater(pdft, max(pdft) * 0.005), pdft, 0.)
            A[:, k] = pdft / linalg.norm(pdft)
            k += 1
    return A


def create_voigt_dict(zfine, mu, Nmu, sigma, Nsigma, Nv, cut=1.e-5):
    """
    Creates a gaussian-voigt dictionary at the same resolution as the original PDF

    :param float zfine: the x-axis for the PDF, the redshift resolution
    :param float mu: [min_mu, max_mu], range of mean for gaussian
    :param int Nmu: Number of values between min_mu and max_mu
    :param float sigma: [min_sigma, max_sigma], range of variance for gaussian
    :param int Nsigma: Number of values between min_sigma and max_sigma
    :param Nv: Number of Voigt profiles per gaussian at given position mu and sigma
    :param float cut: Lower cut for gaussians

    :return: Dictionary as numpy array with shape (len(zfine), Nmu*Nsigma*Nv)
    :rtype: float

    """

    zmid = linspace(mu[0], mu[1], Nmu)
    sig = linspace(sigma[0], sigma[1], Nsigma)
    gamma = linspace(0, 0.5, Nv)
    NA = Nmu * Nsigma * Nv
    Npdf = len(zfine)
    A = zeros((Npdf, NA))
    kk = 0
    for i in xrange(Nmu):
        for j in xrange(Nsigma):
            for k in xrange(Nv):
                #pdft = 1. * exp(-((zfine - zmid[i]) ** 2) / (2.*sig[j]*sig[j]))
                pdft = voigt(zfine, zmid[i], sig[j], sig[j] * gamma[k])
                pdft = where(pdft >= cut, pdft, 0.)
                A[:, kk] = pdft / linalg.norm(pdft)
                kk += 1
    return A


def reconstruct_pdf_v(index, vals, zfine, mu, Nmu, sigma, Nsigma, Nv, cut=1.e-5):
    """
    This function reconstruct the pdf from the indices and values and parameters used to create the dictionary with
    Gaussians and Voigt profiles

    :param int index: List of indices in the dictionary for the selected bases
    :param float vals: values or coefficients corresponding to the listed indices
    :param float zfine: redshift values from the original pdf or used during the sparse representation
    :param float mu: [min_mu, max_mu] values used to create the dictionary
    :param int Nmu: Number of mu values used to create the dictionary
    :param float sigma: [min_sigma, mas_sigma] sigma values used to create the dictionary
    :param int Nsigma: Number of sigma values
    :param int Nv: Number of Voigt profiles used to create dictionary
    :param float cut: cut threshold when creating the dictionary

    :return: the pdf normalized so it sums to one
    """

    zmid = linspace(mu[0], mu[1], Nmu)
    sig = linspace(sigma[0], sigma[1], Nsigma)
    gamma = linspace(0, 0.5, Nv)
    pdf = zeros(len(zfine))
    for kk in xrange(len(index)):
        i = index[kk] / (Nsigma * Nv)
        j = (index[kk] % (Nsigma * Nv)) / Nv
        k = (index[kk] % (Nsigma * Nv)) % Nv
        pdft = voigt(zfine, zmid[i], sig[j], sig[j] * gamma[k])
        pdft = where(pdft >= cut, pdft, 0.)
        pdft = pdft / linalg.norm(pdft)
        pdf += pdft * vals[kk]
        #pdf = where(pdf >= cut, pdf, 0)
    pdf = where(greater(pdf, max(pdf) * 0.005), pdf, 0.)
    if sum(pdf) > 0: pdf = pdf / sum(pdf)
    return pdf


def reconstruct_pdf_int(long_index, header, cut=1.e-5):
    """
    This function reconstruct the pdf from the integer indices only and the parameters used to create the dictionary
    with Gaussians and Voigt profiles

    :param int long_index: List of indices including coefficients (32bits integer array)
    :param dict header: Dictionary of the fits file header with information used to create dictionary and sparse indices
    :param float cut: cut threshold when creating the dictionary

    :return: the pdf normalized so it sums to one
    """

    Ncoef = header['Ncoef']
    zfine = header['z']
    mu = header['mu']
    Nmu = header['Nmu']
    sigma = header['sig']
    Nsigma = header['Nsig']
    Nv = header['Nv']

    VALS = linspace(0, 1, Ncoef)
    dVals = VALS[1] - VALS[0]
    sp_ind = array(map(get_N, long_index))
    spi = sp_ind[:, 0]
    Dind2 = sp_ind[:, 1]
    vals = spi * dVals
    ####
    vals[0]=1.
    ####
    rep_pdf = reconstruct_pdf_v(Dind2, vals, zfine, mu, Nmu, sigma, Nsigma, Nv)
    return rep_pdf


def read_header(fits_file):
    """
    Reads the header from a fits file that stores the sparse indices

    :param str fits_file: Name of fits file
    :return: Dictionary of header to be used to reconstruct PDF
    """

    head = {}
    F = pf.open(fits_file)
    H = F[0].header
    head['Ntot'] = H['N_TOT']
    head['Nmu'] = H['N_MU']
    head['Nsig'] = H['N_SIGMA']
    head['Nv'] = H['N_VOIGT']
    head['Ncoef'] = H['N_COEF']
    head['Nspa'] = H['N_SPARSE']
    head['mu'] = [H['MU1'], H['MU2']]
    head['sig'] = [H['SIGMA1'], H['SIGMA2']]
    head['z'] = F[1].data.field('redshift')
    F.close()
    return head

def get_npeaks(z, pdf):
    """
    Get the number of peaks for a given PDF

    :param float z: the redhisft values of the PDF
    :param float pdf: the values of the PDF

    :return: The number of peaks, positions of the local maximums, local minimums and inflexion points
    """
    local_max = []
    curr = sign(1)
    local_min = []
    local_in = []
    w = where(pdf > 0)[0] #non zeros values
    local_min.append(w[0]) #first non zero value
    for i in xrange(w[0], len(pdf) - 1):
        dy = pdf[i + 1] - pdf[i]
        #if sign(dy)==sign(1) and curr==sign(1) and abs(dy) > 0. and abs(dy) < 1.e-4: local_in.append(i)
        #if sign(dy)==sign(-1) and curr==sign(-1) and abs(dy) > 0. and abs(dy) < 1.e-4: local_in.append(i)
        if sign(dy) == sign(-1) and curr == sign(1):
            local_max.append(i)
        if sign(dy) == sign(1) and curr == sign(-1):
            local_min.append(i)
        if not dy == 0.: curr = sign(dy)
    local_min.append(w[-1]) # last non zero
    N_peak = len(local_max)
    N_peak = min(N_peak, 15) #Up to 15 Gaussians, can be modified
    return N_peak, local_max, local_min, local_in


def initial_guess(z, pdf):
    """
    Computes a initial guess based on local maxima and minima,
    it adds an extra gaussian to the number of peaks
    """
    N_gauss, local_max, local_min, local_in = get_npeaks(z, pdf)
    t0 = []
    w = where(pdf > 0)[0]  #non zeros values
    range_z = max(z[w]) - min(z[w])
    for j in xrange(N_gauss):
        t0.append(pdf[local_max[j]])
        t0.append(z[local_max[j]])
        sigma_approx = (z[local_min[j + 1]] - z[local_min[j]]) / 4.
        t0.append(sigma_approx)
    if len(local_in) > 0:
        for j in xrange(len(local_in)):
            t0.append(pdf[local_in[j]])
            t0.append(z[local_in[j]])
            sigma_approx = (z[1] - z[0]) * 5.
            t0.append(sigma_approx)
            #EXTRA GAUSSIAN
    t0.append(max(pdf) / 2.)
    t0.append(sum(z * pdf))
    t0.append(range_z / 3.)
    return array(t0)


def multi_gauss(P, x):
    """
    Muti-Gaussian function

    :param float P: array with values for amplitud, mean and sigma, P=[A0,mu0,sigma0, A1,mu1, sigma1, ...]
    :param float x: x values

    :return: The multi gaussian
    """
    Ng = int(len(P) / 3)
    p1 = zeros(len(x))
    for i in xrange(Ng):
        p1 += abs(P[0 + i * 3]) * exp(-(x - P[1 + i * 3]) ** 2 / (2. * P[2 + i * 3] * P[2 + i * 3]))
    return p1


def errf(P, x, y):
    """
    Error function to be minimized during fitting
    """
    return y - multi_gauss(P, x)


def fit_multi_gauss(z, pdf, tolerance=1.49e-8):
    """
    Fits a multi gaussian function to the pdf, given a tolerance
    """
    guess = initial_guess(z, pdf)
    Ng = len(guess) / 3
    out_p, pcov = leastsq(errf, guess, args=(z, pdf), ftol=tolerance)
    return out_p


def voigt(x, x_mean, sigma, gamma):
    """
     Voigt profile
     V(x,sig,gam) = Re(w(z)), w(z) Faddeeva function
     z = (x+j*gamma)/(sigma*sqrt(2))

     :param float x: the x-axis values (redshift)
     :param float x_mean: Mean of the gaussian or Voigt
     :param float sigma: Sigma of the original Gaussian when gamma=0
     :param float gamma: Gamma parameter for the Lorentzian profile (Voigt)

     :return: The real values of the Voigt profile at points x
     """

    x = x - x_mean
    z = (x + 1j * gamma) / (sqrt(2.) * sigma)
    It = special.wofz(z).real
    return It


def combine_int(Ncoef, Nbase):
    """
    combine index of base (up to 62500 bases) and value (16 bits integer with sign) in a 32 bit integer
    First half of word is for the value and second half for the index

    :param int Ncoef: Integer with sign to represent the value associated with a base, this is a sign 16 bits integer
    :param int Nbase: Integer representing the base, unsigned 16 bits integer
    :return: 32 bits integer
    """
    return (Ncoef << 16) | Nbase


def get_N(longN):
    """
    Extract coefficients fro the 32bits integer,
    Extract Ncoef and Nbase from 32 bit integer
    return (longN >> 16), longN & 0xffff

    :param int longN: input 32 bits integer

    :return: Ncoef, Nbase both 16 bits integer
    """
    return (longN >> 16), (longN & (2 ** 16 - 1))


def combine3(a, b, c):
    # combine 3 integer,
    # a int8 (0--255) first half
    # b int6 (0--63) after a
    # c int4 (0--3) after b, last 2 bits
    return (a << 8 | (b << 2)) | c


def extract3(d):
    #extract a,b,c (int8,int6,int2) from int16
    return d >> 8, (d & (2 ** 8 - 1)) >> 2, d & (2 ** 2 - 1)
