from numpy import *
import os, sys

path_src = os.path.abspath(os.path.join(os.getcwd(), '../../'))
if not path_src in sys.path: sys.path.insert(1, path_src)
from mlz.ml_codes import *

#X and Y can be anything, in this case SDSS mags and colors for X and photo-z for Y
X = loadtxt('SDSS_MGS.train', usecols=(1, 2, 3, 4, 5, 6, 7), unpack=True).T
Y = loadtxt('SDSS_MGS.train', unpack=True, usecols=(0,))

#this dictionary is optional for this example
#for plotting the color labels
#(automatically included in MLZ)
d = {'u': {'ind': 0}, 'g': {'ind': 1}, 'r': {'ind': 2}, 'i': {'ind': 3}, 'z': {'ind': 4}, 'u-g': {'ind': 5},
     'g-r': {'ind': 6}}

#Calls the Regression Tree mode
T = TPZ.Rtree(X, Y, minleaf=30, mstar=3, dict_dim=d)
T.plot_tree()
#get a list of all branches
branches = T.leaves()
#print first branch, in this case left ,left, left, etc...
print 'branch = ', branches[0]
#print content of branch
content = T.print_branch(branches[0])
print 'branch content'
print content
#get prediction values for a test data (just an example on how to do it)
#using a train objetc
values = T.get_vals(X[10])
print 'predicted values from tree'
print values
print
print 'mean value from prediction', mean(values)
print 'real value', Y[10]
#Note we use a shallow tree and only one tree for example purposes and there
#is a random subsmaple so answer changes every time
