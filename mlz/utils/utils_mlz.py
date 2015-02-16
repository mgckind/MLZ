"""
.. module:: utils_mlz
.. moduleauthor:: Matias Carrasco Kind
"""
__author__ = 'Matias Carrasco Kind'
import numpy
import time
import sys, os
from scipy.interpolate import interp1d as spl
from scipy.integrate import romberg as rom

try:
    from mpi4py import MPI

    PLL = 'MPI'
except:
    PLL = 'SERIAL'

# -----------------------------------------------------------
# Read parameters from input file
# -----------------------------------------------------------
def read_dt_pars(filein, verbose=True, myrank=0):
    """
    Read parameters to be used by the program and convert them into integers/float
    if necessary, returning a class

    :param str filein: name of inputs file, check format here :ref:`input-file`
    :param bool verbose: True or False
    :param int myrank: processor id for multi-core capabilities
    """
    if verbose: printpz('Reading parameters from : ', filein, '...')
    all_keys = allkeys()
    try:
        names, values = numpy.loadtxt(filein, dtype='str', usecols=(0, 2), unpack=True)
    except IOError:
        if myrank == 0:
            printpz_err("Input file \'", filein, "\' not found")
        sys.exit(0)
    except IndexError:
        if myrank == 0:
            printpz_err("One of the Keys in the input file is missing the space between \':\' and its value")
        sys.exit(0)
        # creates dictionary
    DTpars = {}
    sortname = []
    for i, n in enumerate(names):
        if values[i].find(',') != -1:
            val = values[i].split(',')
        else:
            val = values[i]
        if values[i].find('/') != -1 and values[i][-1] != '/': val = values[i] + '/'
        DTpars[n.lower()] = val
        sortname.append(n.lower())
    tofloat = ['nrandom', 'nzbins', 'sigmafactor', 'minz', 'maxz', 'minleaf', 'ntrees', 'natt', 'rmsfactor',
               'ntop', 'iterations', 'alphastart', 'alphaend', 'numbercoef', 'numberbases']
    for key in tofloat:
        try:
            DTpars[key] = float(DTpars[key])
        except KeyError:
            if myrank == 0:
                printpz_err("Key \'", key, "\' not found in input file")
            sys.exit(0)
    for key in all_keys:
        try:
            temp = DTpars[key]
        except KeyError:
            if myrank == 0:
                printpz_err("Key \'", key, "\' not found in input file")
            sys.exit(0)
    DTpars['sname'] = sortname
    DTpars['tofloat'] = tofloat

    class parameters():
        def __init__(self, pars_dict):
            # self.nbzfilename = 'NBZ_' + pars_dict['finalfilename']
            # self.treedictname = 'Tree_Dict_' + pars_dict['finalfilename']
            #self.randomvars = 'Ran_forest_' + pars_dict['finalfilename']
            #self.indexfile = 'Index_' + pars_dict['finalfilename']
            self.oobfraction = 0.333333
            self.minbin = 5
            self.repeatf = 'yes'
            self.treefraction = 0.9
            self.dotrain = 'yes'
            self.dotest = 'yes'
            self.writepdf = 'yes'
            self.all_names = []
            self.all_values = []
            self.sortname = pars_dict['sname']
            self.tofloat = pars_dict['tofloat']
            for k in pars_dict.keys():
                if k == 'sname': continue
                if k == 'tofloat': continue
                com = 'self.' + k + ' = pars_dict[k]'
                self.all_names.append(k)
                self.all_values.append(pars_dict[k])
                exec com
            self.format2 = 'no'
            self.output_names()
            #self.sigmafactor = self.sigmafactor

        def output_names(self):
            self.randomcatname = 'Random_Cat_' + self.finalfilename
            self.treefilename = 'Altrees_' + self.finalfilename
            self.somfilename = 'SOM_' + self.finalfilename

        def print_all(self):
            for name in self.all_names:
                print '.' + name
            self.path_results = ''

    DT_p = parameters(DTpars)
    DT_p.path_results = DT_p.path_output + 'results/'
    DT_p.path_output_trees = DT_p.path_output + 'trees/'
    DT_p.path_output_maps = DT_p.path_output + 'maps/'
    if DT_p.importancefile[-1] == '/': DT_p.importancefile = DT_p.importancefile[:-1]
    if DT_p.predictionmode != 'TPZ' and DT_p.predictionmode != 'SOM' and DT_p.predictionmode != 'TPZ_C' and DT_p.predictionmode != 'BPZ':
        if myrank == 0:
            printpz_err("Key PredictionMode is wrong")
        sys.exit(0)
    if DT_p.predictionclass != 'Reg' and DT_p.predictionclass != 'Class':
        if myrank == 0:
            printpz_err("Key PredictionClass is wrong")
        sys.exit(0)
    return DT_p


class tcolor:
    yellow = '\033[93m'
    green = '\033[92m'
    off = '\033[0m'
    red = '\033[91m'
    purple = '\033[95m'
    blue = '\033[94m'
    on_yellow = '\033[43m'


def printpz(*args, **kwargs):
    title = '[MLZ]  '
    verb = True
    red = False
    green = False
    blue = False
    yellow = False
    purple = False
    if kwargs.has_key('verb'): verb = kwargs['verb']
    if kwargs.has_key('red'): red = kwargs['red']
    if kwargs.has_key('green'): green = kwargs['green']
    if kwargs.has_key('blue'): blue = kwargs['blue']
    if kwargs.has_key('yellow'): yellow = kwargs['yellow']
    if kwargs.has_key('purple'): purple = kwargs['purple']
    if len(args) == 0:
        pass
    else:
        for stt in args:
            title += str(stt)
    if verb:
        if red:
            print tcolor.red + title + tcolor.off
        elif green:
            print tcolor.green + title + tcolor.off
        elif blue:
            print tcolor.blue + title + tcolor.off
        elif yellow:
            print tcolor.yellow + title + tcolor.off
        elif purple:
            print tcolor.purple + title + tcolor.off
        else:
            print title


def printpz_err(*args):
    printpz()
    printpz(''' ===================== ''', red=True)
    printpz(''' *       ERROR       * ''', red=True)
    printpz(''' ===================== ''', red=True)
    if len(args) != 0: arg2 = [i for i in args] + [" <<<"]
    printpz(">>> ", *arg2)
    printpz()
    printpz('Exiting program')
    printpz()
    print


def percentile(Nvals, percent):
    """
    Find the percentile of a list of values.
    :param float Nvals: list of values
    :param float percent: a percentile value between 0 and 1.
    :return: percentile value
    :rtype: float
    """
    Nvals = Nvals[numpy.argsort(Nvals)]
    kper = (len(Nvals) - 1) * percent
    f = numpy.floor(kper)
    c = numpy.ceil(kper)
    if f == c:
        return float(Nvals[int(kper)])
    d0 = float(Nvals[int(f)]) * (c - kper)
    d1 = float(Nvals[int(c)]) * (kper - f)
    return d0 + d1


# ------------------------------------------------------------------
class bias:
    """
    Creates a instance to compute some metrics for the photo-z calculation for quick analysis

    :param float zs: Spectrocopic redshift  array
    :param float zb: Photometric redshift
    :param str name: name for identification
    :param float zmin: Minimum redshift for binning
    :param float zmax: Maximum redshift for binning
    :param int nbins: Number of bins used
    :param int mode: 0 (binning in spec-z) or 1 (binning in photo-z)
    :param function d_z: function to be applied on z_phot and z_spec, default (z_phot-z_spec)
    :param bool verb: verbose?
    """

    def __init__(self, zs, zb, name, zmin, zmax, nbins, mode=1, d_z=lambda x, y: x - y, verb=True):
        zbins = numpy.linspace(zmin, zmax, nbins + 1)
        zmid = 0.5 * (zbins[1:] + zbins[:-1])
        self.bins = zmid
        self.zs = zs
        self.zb = zb
        self.name = name
        self.mean = numpy.zeros(len(zmid))
        self.absmean = numpy.zeros(len(zmid))
        self.abssigma = numpy.zeros(len(zmid))
        self.outlier = numpy.zeros(len(zmid))
        self.median = numpy.zeros(len(zmid))
        self.sigma = numpy.zeros(len(zmid))
        self.mean_e = numpy.zeros(len(zmid))
        self.sigma68 = numpy.zeros(len(zmid))
        self.frac2 = numpy.zeros(len(zmid))
        self.frac3 = numpy.zeros(len(zmid))
        top = self.name + ' : %d galaxies' % len(zs)
        if verb: printpz(top)
        for i in xrange(len(zmid)):
            if mode == 0: wt = numpy.where((zs >= zbins[i]) & (zs < zbins[i + 1]))
            if mode == 1: wt = numpy.where((zb >= zbins[i]) & (zb < zbins[i + 1]))
            deltaz = d_z(zb, zs)
            if numpy.shape(wt)[1] == 0: continue
            tempz = numpy.sort(deltaz[wt])
            self.sigma68[i] = 0.5 * (percentile(tempz, 0.84) - percentile(tempz, 0.16))
            self.mean[i] = numpy.mean(deltaz[wt])
            self.absmean[i] = numpy.mean(abs(deltaz[wt]))
            self.abssigma[i] = numpy.std(abs(deltaz[wt]))
            self.median[i] = numpy.median(deltaz[wt])
            self.sigma[i] = numpy.std(deltaz[wt])
            w2 = numpy.where(abs(deltaz[wt] - self.mean[i]) > 2 * self.sigma[i])
            w3 = numpy.where(abs(deltaz[wt] - self.mean[i]) > 3 * self.sigma[i])
            wout = numpy.where(abs(deltaz[wt]) > 0.1)
            self.frac2[i] = 1. * numpy.shape(w2)[1] / (1. * numpy.shape(wt)[1])
            self.frac3[i] = 1. * numpy.shape(w3)[1] / (1. * numpy.shape(wt)[1])
            self.outlier[i] = 1. * numpy.shape(wout)[1] / (1. * numpy.shape(wt)[1])


class conf:
    """
    Computes confidence level (zConf) as a function of photometric redshift

    :param float zconf: zConf array for galaxies
    :param float zb: Photometric redshifts
    :param float zmin: Minimum redshift for binning
    :param float zmax: Maximum redshift for binning
    :param int nbins: Number of bins used
    """

    def __init__(self, zconf, zb, zmin, zmax, nbins):
        zbins = numpy.linspace(zmin, zmax, nbins + 1)
        zmid = 0.5 * (zbins[1:] + zbins[:-1])
        self.bins = zmid
        self.zC = numpy.zeros(len(zmid))
        self.zC_e = numpy.zeros(len(zmid))
        for i in xrange(len(zmid)):
            wt = numpy.where((zb >= zbins[i]) & (zb < zbins[i + 1]))
            if numpy.shape(wt)[1] == 0: continue
            self.zC[i] = numpy.mean(zconf[wt])
            self.zC_e[i] = numpy.std(zconf[wt])


def zconf_dist(conf, nbins):
    """
    Computes the distribution of Zconf for different bins between 0 and 1

    :param float conf: zConf values
    :param int nbins: number of bins
    :return: zConf dist, bins
    :rtype: float,float
    """
    bins = numpy.linspace(0., 1, nbins)
    s_conf = numpy.sort(conf)
    z_conf = numpy.zeros(len(bins))
    for i in xrange(len(bins)):
        z_conf[i] = percentile(s_conf, bins[i])
    return z_conf, bins


class Stopwatch:
    """
    Stopwatch and some time procedures

    :param str verb: 'yes' or 'no' (verbose?)
    """

    def __init__(self, verb='yes'):
        verbin = True
        if verb == 'no': verbin = False
        self.start0 = time.time()
        self.restart(verb='Start', verbose=verbin)

    def elapsed(self, only_sec=False, verbose=True):
        """
        Prints and saves elapsed time

        :param bool only_sec: set this to True for the elapsed time prints in seconds only
        :param bool verbose: Prints on screen
        """
        self.count += 1
        self.end = time.time()
        if verbose:
            printpz()
            printpz('-----------------time--------------------')
            printpz('Check ' + str(self.count) + ' : ', time.ctime(self.end))
        if only_sec:
            if verbose: printpz('Elapsed ' + str(self.count) + ' [seconds] : ', self.end - self.start)
        else:
            if verbose: printpz('Elapsed ' + str(self.count) + ' :',
                                time.strftime('%H:%M:%S', time.gmtime(self.end - self.start)))
        if verbose:
            printpz('-----------------time--------------------')
            printpz()

    def restart(self, verb='Restart', verbose=True):
        """
        Set the counter to zero, keeps tracking of starting time

        :param str verb: 'Start' (default) or 'Restart' (set the counter to zero and the starting
        time to current time, keeps the initial starting in self.start0)
        """
        self.count = 0
        if verb == 'Restart':
            self.start = time.time()
        if verb == 'Start':
            self.start0 = time.time()
            self.start = self.start0
        if verbose:
            printpz
            printpz('-----------------time--------------------')
            printpz(verb + ' : ', time.ctime(self.start))
            if verb == 'Restart':
                printpz('Elapsed from Start :', time.strftime('%H:%M:%S', time.gmtime(self.start - self.start0)))
            printpz('-----------------time--------------------')
            printpz()


def print_dtpars(DTpars, outfile, system=False):
    """
    Prints the values from class Pars to a file

    :param class DTpars: class Pars from input file
    :param str outfile: output filename
    """
    names = numpy.array(DTpars.all_names)
    vals = list(DTpars.all_values)
    sname = DTpars.sortname
    if not system: fo = open(outfile, 'w')
    for k in sname:
        w = numpy.where(names == k)[0]
        w = w[0]
        if type(vals[w]) == type([1, 2, 3]):
            svals = ','.join(vals[w])
        else:
            svals = vals[w]
        line = "%-20s : %s \n" % (names[w], svals)
        if system:
            printpz(line.strip())
        else:
            fo.write(line)
    if not system: fo.close()


def get_area(z, pdf, z1, z2):
    """
    Compute area under photo-z Pdf between z1 and z2, PDF must add to 1

    :param float z: redshift
    :param float pdf: photo-z PDF
    :param float z1: Lower boundary
    :param float z2: Upper boundary
    :return: area between z1 and z2
    :rtype: float
    """
    PP = spl(z, pdf, bounds_error=False, fill_value=0.0)
    area = rom(PP, z1, z2, tol=1.0e-05, rtol=1.0e-05)
    dz = z[1] - z[0]
    return area / dz


def get_probs(z, pdf, z1, z2):
    pdf = pdf / numpy.sum(pdf)
    PP = spl(z, pdf, bounds_error=False, fill_value=0.0)
    dzo = z[1] - z[0]
    dz = 0.001
    Ndz = int((z2 - z1) / dz)
    A = 0
    for i in xrange(Ndz):
        A += dz * PP((z1) + dz / 2. + dz * i)
    return A / dzo


def get_prob_Nz(z, pdf, zbins):
    pdf = pdf / numpy.sum(pdf)
    PP = spl(z, pdf, bounds_error=False, fill_value=0.0)
    dzo = z[1] - z[0]
    dz = 0.001
    Ndz = int((zbins[1] - zbins[0]) / dz)
    Nzt = numpy.zeros(len(zbins) - 1)
    for j in xrange(len(Nzt)):
        for i in xrange(Ndz):
            Nzt[j] += dz * PP((zbins[j]) + dz / 2. + dz * i)
    return Nzt / dzo


def inbin(z1, minz, maxz, Nbins):
    dz = (maxz - minz) / (1. * Nbins)
    i1 = numpy.floor((z1 - minz) / dz)
    i1 = max(i1, 0)
    i1 = min(i1, Nbins - 1)
    return int(i1)


def compute_A(z, pdf, za, zb):
    # Also computes area but using midpoint rule
    w = numpy.where((z >= za) & (z <= zb))[0]
    A = numpy.sum(pdf[w])
    return A * 1.


def compute_error(z, pdf, zv):
    """
    Computes the error in the PDF calculation using a reference values from PDF
    it computes the 68% percentile limit around this value

    :param float z: redshift
    :param float pdf: photo-z PDF
    :param float zv: Reference value from PDF (can be mean, mode, median, etc.)
    :return: error associated to reference value
    :rtype: float
    """
    res = 0.001
    PP = spl(z, pdf, bounds_error=False, fill_value=0.0)
    dz = z[1] - z[0]
    j = 0
    area = 0
    while area <= 0.68:
        j += 1
        za = zv - res * j
        zb = zv + res * j
        area = rom(PP, za, zb, tol=1.0e-04, rtol=1.0e-04) / dz
    return j * res


def compute_error2(z, pdf, zv):
    L1 = 0.0001
    L2 = (max(z) - min(z)) / 2.
    PP = spl(z, pdf, bounds_error=False, fill_value=0.0)
    dz = z[1] - z[0]
    eps = 0.05
    za1 = zv - L1
    zb1 = zv + L1
    area = 0
    LM = L2
    while abs(area - 0.68) > eps:
        za2 = zv - LM
        zb2 = zv + LM
        area = rom(PP, za2, zb2, tol=1.0e-04, rtol=1.0e-04) / dz
        Lreturn = LM
        if area > 0.68:
            L2 = LM
            LM = (L1 + L2) / 2.
        else:
            L1 = LM
            LM = (L1 + L2) / 2.
    return Lreturn


def compute_error3(z, pdf, zv):
    dz = z[1] - z[0]
    ib = numpy.argmin(abs(zv - z))
    area = pdf[ib]
    nm = len(z) - 1
    j = 0
    i2 = ib + 1
    i1 = ib
    if sum(pdf) < 0.00001 : return 9.99
    while area <= 0.68:
        area1 = sum(pdf[i1:i2])
        e681 = dz * (i2 - i1)
        j += 1
        i1 = max(0, ib - j)
        i2 = min(nm, ib + j) + 1
        area = sum(pdf[i1:i2])
        e68 = dz * (i2 - i1)
        ef = ((e68 - e681) / (area - area1)) * (0.68 - area1) + e681
    return ef / 2.


def compute_zConf(z, pdf, zv, sigma):
    """
    Computes the confidence level of the pdf with respect a reference value
    as the area between zv-sigma(1+zv) and zv+sigma(1+zv)

    :param float z: redshift
    :param float pdf: photo-z PDF
    :param float zv: reference value
    :param float sigma: extent of confidence
    :return: zConf
    :rtype: float
    """
    z1a = zv - sigma * (1. + zv)
    z1b = zv + sigma * (1. + zv)
    zC1 = get_area(z, pdf, z1a, z1b)
    return zC1


def compute_zConf2(z, pdf, zv, sigma):
    z1a = zv - sigma * (1. + zv)
    z1b = zv + sigma * (1. + zv)
    ib1 = numpy.argmin(abs(z1a - z))
    ib2 = numpy.argmin(abs(z1b - z)) + 1
    return sum(pdf[ib1:ib2])


def get_limits(ntot, Nproc, rank):
    """
    Get limits for farming an array to multiple processors

    :param int ntot: Number of objects in array
    :param int Nproc: number of processor
    :param int rank: current processor id
    :return: L1,L2 the limits of the array for given processor
    :rtype: int, int
    """
    jpproc = numpy.zeros(Nproc) + int(ntot / Nproc)
    for i in xrange(Nproc):
        if (i < ntot % Nproc): jpproc[i] += 1
    jpproc = map(int, jpproc)
    st = rank
    st = sum(jpproc[:rank]) - 1
    s0 = int(st + 1)
    s1 = int(st + jpproc[rank]) + 1
    return s0, s1


def print_welcome():
    bla = os.system('clear')
    print
    printpz('''------------------------------------------------''', blue=True)
    printpz('''|      ____    ____    _____      ________     |''', blue=True)
    printpz('''|     |_   \  /   _|  |_   _|    |  __   _|    |''', blue=True)
    printpz('''|       |   \/   |      | |      |_/  / /      |''', blue=True)
    printpz('''|       | |\  /| |      | |   _     .'.' _     |''', blue=True)
    printpz('''|      _| |_\/_| |_    _| |__/ |  _/ /__/ |    |''', blue=True)
    printpz('''|     |_____||_____|  |________| |________|    |''', blue=True)
    printpz('''|                                              |''', blue=True)
    printpz('''|                                              |''', blue=True)
    printpz('''|      Machine   Learning   for   photo-Z      |''', blue=True)
    printpz('''|----------------------------------------------|''', blue=True)
    printpz()


def print_mode(mode):
    printpz()
    printpz('''-----------------------------------------------------------''')
    if mode == 'Reg':
        printpz(''' __   ___  __   __   ___  __   __     __      ''')
        printpz('''|__) |__  / _` |__) |__  /__` /__` | /  \ |\ |''')
        printpz('''|  \ |___ \__> |  \ |___ .__/ .__/ | \__/ | \|''')
        printpz('''-----------------------------------------------------------''')
    if mode == 'Class':
        printpz(''' __             __   __     ___    __       ___    __      ''')
        printpz('''/  ` |     /\  /__` /__` | |__  | /  `  /\   |  | /  \ |\ |''')
        printpz('''\__, |___ /~~\ .__/ .__/ | |    | \__, /~~\  |  | \__/ | \|''')
        printpz('''-----------------------------------------------------------''')
    if mode == 'SOM':
        printpz(''' __   __       ''')
        printpz('''/__` /  \  |\/|''')
        printpz('''.__/ \__/  |  |''')
        printpz('''-----------------------------------------------------------''')
        printpz()
    if mode == 'TPZ':
        printpz('''___  __  __ ''')
        printpz(''' |  |__)  / ''')
        printpz(''' |  |    /_ ''')
        printpz('''-----------------------------------------------------------''')
        printpz()
    if mode == 'TPZ_C':
        printpz('''___  __  __      __  ''')
        printpz(''' |  |__)  /     /  ` ''')
        printpz(''' |  |    /_ ___ \__, ''')
        printpz('''-----------------------------------------------------------''')
        printpz()
    if mode == 'BPZ':
        printpz(''' __   __  __ ''')
        printpz('''|__) |__)  / ''')
        printpz('''|__) |    /_ ''')
        printpz('''-----------------------------------------------------------''')
        printpz()


def usage():
    printpz()
    printpz('Usage :: ')
    printpz('------------------------------------------')
    printpz('./runMLZ <inputs file>')
    printpz('------------------------------------------')
    printpz()
    printpz('''See http://lcdm.astro.illinois.edu/static/code/mlz/MLZ-1.1/doc/html/index.html for more info''')


def allkeys():
    keys_input = [ \
        'trainfile            ', \
        'testfile             ', \
        'finalfilename        ', \
        'path_train           ', \
        'path_test            ', \
        'path_output          ', \
        'columns              ', \
        'att                  ', \
        'columns_test         ', \
        'keyatt               ', \
        'checkonly            ', \
        'predictionmode       ', \
        'predictionclass      ', \
        'nrandom              ', \
        'minz                 ', \
        'maxz                 ', \
        'multiplefiles        ', \
        'nzbins               ', \
        'minleaf              ', \
        'impurityindex        ', \
        'ntrees               ', \
        'natt                 ', \
        'ooberror             ', \
        'varimportance        ', \
        'sigmafactor          ', \
        'topology             ', \
        'periodic             ', \
        'ntop                 ', \
        'iterations           ', \
        'somtype              ', \
        'alphastart           ', \
        'alphaend             ', \
        'importancefile       ', \
        'sparserep            ', \
        'sparsedims           ', \
        'numbercoef           ', \
        'numberbases          ', \
        'originalpdffile      ', \
        'writefits            ', \
        'rmsfactor            ']
    keys_input = [aa.strip() for aa in keys_input]
    return keys_input















