"""
.. module:: plotting
.. moduleauthor:: Matias Carrasco Kind
"""
__author__ = 'Matias Carrasco Kind'
from numpy import *
import os, sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import random as rn
import pyfits as pf

path_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if not path_src in sys.path: sys.path.insert(1, path_src)
from ml_codes import *
from utils import *


class Qplot():
    """
    Creates a qplot instance to produce a set of useful plot for quick
    analysis

    :param str inputs_file: path to input file where all information and parameters are declared :ref:`input-file`
    """

    def __init__(self, inputs_file):
        utils_mlz.print_welcome()
        Pars_in = utils_mlz.read_dt_pars(inputs_file, verbose=True)
        self.Pars = Pars_in
        self.inputs_file = inputs_file

    def plot_tree(self, ntree=0, save_files='no', fileroot='TPZ', path=''):
        """
        Plot a tree created during the training process,
        uses the Graphviz package (dot and neato)

        :param int ntree: Number of created tree, default is 0
        :param str save_files: Saves the created files from Graphviz (the .png and the .dot files) 'yes'/'no'
        :param str fileroot: root name for saved files
        :param str path: path name for output files
        """
        path_tree = self.Pars.path_output_trees
        fileT = self.Pars.treefilename + '_%04d.npy' % ntree
        utils_mlz.printpz('Plotting ', fileT)
        T = load(path_tree + fileT)
        T = T.item()
        if T.dict_dim == 'all':
            dd = {}
            for i in xrange(len(self.Pars.att)):
                dd[self.Pars.att[i]] = {'ind': i}
        T.dict_dim = dd
        T.plot_tree(save_png=save_files, fileout=fileroot, path=path)

    def plot_map(self, nmap=0, colbar='yes', min_m=-100, max_m=-100):
        """
        Plot a map created during the training process,

        :param int nmap: Number of created map, default is 0
        :param float min_m: Lower limit for coloring the cells, -100 uses min value
        :param float max_m: Upper limit for coloring the cells, -100 uses max value
        :param str colbar: Include a colorbar ('yes','no')
        """

        path_map = self.Pars.path_output_maps
        fileM = self.Pars.somfilename + '_%04d.npy' % nmap
        utils_mlz.printpz('Plotting ', fileM)
        M = load(path_map + fileM)
        M = M.item()
        M.plot_map(colbar=colbar, min_m=min_m, max_m=max_m)

    def plot_importance(self, result_id=0, Nzb=10, list_att=''):
        """
        Plot ranking of importance of attributes used during the training/testing process

        .. note ::

            The key `OobError` and `VarImportance` in :ref:`input-file` must be set to 'yes' to compute these quantities

        :param int results_id: Result id number as the output on the results folder, default 0
        :param int Nzb: Number of redshift bins
        """
        filenum = str(result_id)
        froot = self.Pars.path_results + self.Pars.finalfilename + '_oob'
        zs, za, zb, oa, ob, ea, eb = loadtxt(froot + '.' + filenum + '.mlz', unpack=True)
        fdz = lambda x, y: (abs(x - y)) / (1. + y)
        O_all = utils_mlz.bias(zs, zb, 'OOB_all', self.Pars.minz, self.Pars.maxz, 1,
                               mode=0, d_z=fdz, verb=False)
        ki = arange(len(self.Pars.att))
        Im0 = zeros(len(ki))
        Im1 = zeros(len(ki))
        keys = []
        utils_mlz.printpz("Importance")
        utils_mlz.printpz("----------")
        for k in xrange(len(self.Pars.att)):
            kk = self.Pars.att[k]
            keys.append(kk)
            filev = froot + '_' + kk + '.' + filenum + '.mlz'
            zs, za, zb, oa, ob, ea, eb = loadtxt(filev, unpack=True)
            O_temp = utils_mlz.bias(zs, zb, 'OOB_temp', self.Pars.minz, self.Pars.maxz, 1,
                                    mode=0, d_z=fdz, verb=False)
            Im0[k] = O_temp.mean / O_all.mean
            Im1[k] = O_temp.sigma / O_all.sigma
            del O_temp
            imp = '%.5f' % Im1[k]
            utils_mlz.printpz(kk, ' ', imp)

        fimportance = froot + '.' + filenum + '.importance'
        FF = open(fimportance, 'w')
        linew = ','.join(keys)
        linew = '#' + linew + '\n'
        FF.write(linew)
        for i in xrange(len(ki)):
            linew = '%.8f\n' % Im0[i]
            FF.write(linew)
        FF.close()

        utils_mlz.printpz()
        utils_mlz.printpz("Importance file saved in: ", fimportance)

        Im = (Im0 + Im1) / 2.
        sk = argsort(Im0)
        sk = sk[::-1]
        fig, ax = plt.subplots()
        ax.plot(ki, Im0[sk], 'bo-', label='bias')
        ax.plot(ki, Im1[sk], 'go-', label='sigma')
        ax.plot(ki, Im[sk], 'ro-', label='avg')
        ax.set_xlabel('Importance', fontsize=16)
        ax.set_ylabel('Attributes', fontsize=16)
        plt.legend(loc=0)
        ax.set_xticks(ki)
        ax.set_xticklabels(array(keys)[sk], rotation=40, ha='right', fontsize=12)
        ax.set_xlim(-1, len(ki))

        ##
        zs, za, zb, oa, ob, ea, eb = loadtxt(froot + '.' + filenum + '.mlz', unpack=True)
        O_all = utils_mlz.bias(zs, zb, 'OOB_all', self.Pars.minz, self.Pars.maxz, Nzb,
                               mode=1, d_z=fdz, verb=False)
        ki = arange(len(self.Pars.att))
        Im0 = zeros((Nzb, len(ki)))
        Im1 = zeros((Nzb, len(ki)))
        keys = []
        for k in xrange(len(self.Pars.att)):
            kk = self.Pars.att[k]
            keys.append(kk)
            filev = froot + '_' + kk + '.' + filenum + '.mlz'
            zs, za, zb, oa, ob, ea, eb = loadtxt(filev, unpack=True)
            O_temp = utils_mlz.bias(zs, zb, 'OOB_temp', self.Pars.minz, self.Pars.maxz, Nzb,
                                    mode=1, d_z=fdz, verb=False)
            Im0[:, k] = O_temp.mean / O_all.mean
            Im1[:, k] = O_temp.sigma / O_all.sigma
            del O_temp

        fig2, ax2 = plt.subplots()
        if list_att=='':
            ax2.plot(O_all.bins, (Im0[:, sk[0]] + Im1[:, sk[0]]) / 2., 'bo-', label=array(keys)[sk[0]])
            ax2.plot(O_all.bins, (Im0[:, sk[1]] + Im1[:, sk[1]]) / 2., 'go-', label=array(keys)[sk[1]])
            ax2.plot(O_all.bins, (Im0[:, sk[-1]] + Im1[:, sk[-1]]) / 2., 'ro-', label=array(keys)[sk[-1]])
        else:
            for katt in list_att:
                wk=where(array(keys)==katt)[0]
                ax2.plot(O_all.bins, (Im0[:, wk[0]] + Im1[:, wk[0]]) / 2., 'o-', label=array(keys)[wk[0]])
        ax2.set_xlabel('redshift', fontsize=16)
        ax2.set_ylabel('Importance', fontsize=16)
        plt.legend(loc=0)
        plt.show()

    def plot_results(self, result_1=0, zconf_1=0., result_2=0, zconf_2=0.):
        """
        Plots a summary of main results for photometric redshifts, it has user interactive plots.

        :param int result_1: result id (run number) as appears on the results , default = 0 (uses mean of PDF for metrics)
        :param float zconf_1: confidence level cut for file 1
        :param int result_2: result id (run number) as appears on the results folder for a second optional file , default
            shows file 1 instead using the mode for the metrics
        :param float zconf_2: confidence level cut for file 2
        """

        arg1 = str(result_1)
        arg2 = str(zconf_1)
        arg3 = str(result_2)
        arg4 = str(zconf_2)

        if result2 == 0:
            os.system("python plotting/plot_results.py " + self.inputs_file + " " + arg1 + " " + arg2)
        else:
            os.system(
                "python plotting/plot_results.py " + self.inputs_file + " " + arg1 + " " + arg2 + " " + arg3 + " " + arg4)

    def plot_pdf_use(self, result_id=0):
        """
        PLots the redshift distribution using PDFs and using one single estimator and a map of zphot vs zspec using also
        PDFs.

        .. note::
            The code utils/use_pdfs must be run first in order to create the needed files,
            it can be run in parallel

        :param int result_id: result id (run number) as appears on the results , default = 0
        """
        ir = result_id
        path_results = self.Pars.path_results
        filebase = self.Pars.finalfilename
        H1 = load(path_results + filebase + '.' + str(ir) + '_map.npy')
        minz = self.Pars.minz
        maxz = self.Pars.maxz
        lev = 10
        maxH = H1.max()
        minH = maxH / 20.
        LL = linspace(minH, maxH, lev)
        plt.contourf(H1, levels=LL, extent=[minz, maxz, minz, maxz], origin='lower', cmap=cm.jet)
        plt.colorbar()
        plt.plot([minz, maxz], [minz, maxz], 'r--', lw=1.5)
        plt.xlabel(r'$z_{spec}$', fontsize=15)
        plt.ylabel(r'$z_{phot}$', fontsize=15)
        plt.title('Using photo-z PDF')

        plt.figure()

        zz, nn = loadtxt(path_results + filebase + '.' + str(ir) + '_zdist', unpack=True)
        zs, za, zb, oa, ob, ea, eb = loadtxt(path_results + filebase + '.' + str(ir) + '.mlz', unpack=True)
        Nbins = len(nn)

        Nz = linspace(minz, maxz, Nbins + 1)
        Nzmid = 0.5 * (Nz[1:] + Nz[:-1])

        ns, zsp = histogram(zs, bins=Nz, normed=True)
        na, zap = histogram(za, bins=Nz, normed=True)
        nb, zbp = histogram(zb, bins=Nz, normed=True)
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 2])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        fzspec = ns / sum(ns)
        fzphot = nn / sum(nn)
        fzmode = na / sum(na)
        fzmean = nb / sum(nb)

        ax1.plot(zz, fzspec, color='black', label='zspec')
        ax1.fill_between(zz, fzspec, color='gray', label='zspec', alpha=0.7)
        ax1.plot(zz, fzmean, lw=2, label='PDF mean', color='blue')
        ax1.plot(zz, fzmode, lw=2, label='PDF mode', color='green')
        ax1.plot(zz, fzphot, lw=2, label='stacked PDF', color='red')
        ax1.set_xlabel('redshift')
        ax1.set_ylabel('$N(z)$', fontsize=16)
        ax1.set_xlim(minz, maxz)
        ax1.legend()

        ax2.plot(zz, abs(fzspec - fzmean), color='blue', lw=1.2)
        ax2.plot(zz, abs(fzspec - fzmode), color='green', lw=1.2)
        ax2.plot(zz, abs(fzspec - fzphot), color='red', lw=1.2)
        ax2.set_xlabel('redshift')
        ax2.set_ylabel('$|\Delta N(z)$|', fontsize=16)
        ax2.set_xlim(minz, maxz)

        plt.show()

    def plot_sparse(self, result_id=0, kgal=-1):
        """
        Plot original and sparse representation of a random select galaxy

        .. note ::

            Both the original and the spare rep. files must exist

        :param int results_id: Result id number as the output on the results folder, default 0
        :param int kgal: Id for specific galaxy
        """

        filenum = str(result_id)
        froot = self.Pars.path_results + self.Pars.finalfilename
        if self.Pars.multiplefiles == 'yes':
            if self.Pars.writefits == 'no': forig = froot + '.' + filenum + '.P_0.npy'
            if self.Pars.writefits == 'yes':forig = froot + '.' + filenum + '.P_0.fits'
            ffits = froot + '.' + filenum + '.Psparse_0.fits'
        else:
            if self.Pars.writefits == 'no' : forig = froot + '.' + filenum + '.P.npy'
            if self.Pars.writefits == 'yes' : forig = froot + '.' + filenum + '.P.fits'
            ffits = froot + '.' + filenum + '.Psparse.fits'

        if self.Pars.writefits == 'no' : PO = load(forig)
        if self.Pars.writefits == 'yes' :
            Temp=pf.open(forig)
            PO=Temp[1].data.field('PDF values')
            Temp.close()

        F = pf.open(ffits)
        P = F[2].data.field('Sparse_indices')
        F.close()

        if kgal < 0:
            k = rn.sample(xrange(len(PO) - 1), 1)[0]
        else:
            k = kgal

        head = pdf_storage.read_header(ffits)
        z = head['z']

        rep_pdf = pdf_storage.reconstruct_pdf_int(P[k], head)

        plt.figure()
        plt.plot(z, PO[k] / sum(PO[k]), label='original')
        plt.plot(z, rep_pdf, label='Sparse rep')
        plt.xlabel('redshift')
        plt.ylabel('P(z)')
        plt.legend(loc=0)
        title = 'Galaxy example No: %d out of %d' % (k, len(P))
        plt.title(title)

        plt.figure()
        AD = pdf_storage.create_voigt_dict(z, head['mu'], head['Nmu'], head['sig'], head['Nsig'], head['Nv'])
        sp_ind = array(map(pdf_storage.get_N, P[k]))
        spi = sp_ind[:, 0]
        Dind2 = sp_ind[:, 1]
        AA = linspace(0, 1, head['Ncoef'])
        Da = AA[1] - AA[0]
        vals = spi * Da
        delta = zeros(shape(AD)[1])
        delta[Dind2] = vals

        pdfr = dot(AD, delta)
        plt.plot(z, pdfr / sum(pdfr), 'r-', lw=2, label='Sparse Rep.')
        for i in xrange(len(Dind2)):
            plt.plot(z, AD[:, Dind2[i]] * vals[i] / sum(pdfr), 'k-')
        plt.plot(z, AD[:, Dind2[0]] * vals[0] / sum(pdfr), 'k-', label='bases')
        plt.xlabel('redshift')
        plt.ylabel('P(z)')
        plt.legend(loc=0)
        title = 'Galaxy example No: %d out of %d' % (k, len(P))
        plt.title(title)

        plt.show()




