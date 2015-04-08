import sys

files_f90 = ['mlz/ml_codes/som.f90', ]
from numpy.distutils.core import setup, Extension

extra_link_args = []
libraries = []
library_dirs = []
include_dirs = ['mlz/ml_codes']
setup(
    name='MLZ',
    version='1.2',
    author='Matias Carrasco Kind',
    author_email='mcarras2@illinois.edu',
    ext_modules=[Extension('somF', files_f90, include_dirs=['mlz/ml_codes'], ), ],
    packages=['mlz', 'mlz.plot', 'mlz.utils', 'mlz.test', 'mlz.ml_codes'],
    py_modules=['mlz','mlz.plot','mlz.utils','mlz.test','mlz.ml_codes'],
    package_data={'mlz': ['test/SDSS_MGS.train', 'test/SDSS_MGS.test', 'test/SDSS_MGS.inputs', 'plot/*.txt']},
    scripts=['mlz/runMLZ', 'mlz/plot/plot_map', 'mlz/plot/plot_results', 'mlz/plot/plot_importance',
             'mlz/plot/plot_tree', 'mlz/utils/use_pdfs', 'mlz/plot/plot_pdf_use', 'mlz/plot/plot_sparse'],
    license='LICENSE.txt',
    description='MLZ: Machine Learning for photo-Z, a photometric redshift PDF estimator',
    long_description=open('README.txt').read(),
    url='http://lcdm.astro.illinois.edu/static/code/mlz/MLZ-1.2/doc/html/index.html',
    install_requires=['mpi4py', 'numpy', 'matplotlib', 'healpy', 'scipy', 'pyfits'],
)
