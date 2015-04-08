from numpy import *
import os, sys

path_src = os.path.abspath(os.path.join(os.getcwd(), '../../'))
if not path_src in sys.path: sys.path.insert(1, path_src)
from mlz.ml_codes import *

#X and Y can be anything, in this case SDSS mags and colors for X and photo-z for Y
X = loadtxt('SDSS_MGS.train', usecols=(1, 2, 3, 4, 5, 6, 7), unpack=True).T
Y = loadtxt('SDSS_MGS.train', unpack=True, usecols=(0,))


#Calls the SOMZ mode
M = SOMZ.SelfMap(X,Y,Ntop=15,iterations=100,periodic='yes')
#creates a map
M.create_mapF()
#evaluates it with the Y entered, or anyoher desired colum
M.evaluate_map()
#plots the map
M.plot_map()
#get prediction values for a test data (just an example on how to do it)
#using a train objetc
values = M.get_vals(X[10])
print
print 'mean value from prediction (hex)', mean(values)
print 'real value', Y[10]
#Note we use a low-resoution map and only one map for example purposes
#evaluate other column, for example the 'g' magnitude
M.evaluate_map(inputY=X[:,1])
M.plot_map()


#Try other topology
M = SOMZ.SelfMap(X,Y,topology='sphere',Ntop=4,iterations=100,periodic='yes')
#creates a map
M.create_mapF()
#evaluates it with the Y entered, or anyoher desired colum
M.evaluate_map()
#plots the map
M.plot_map()
#get prediction values for a test data (just an example on how to do it)
#using a train objetc
values = M.get_vals(X[10])
print
print 'mean value from prediction (sphere)', mean(values)
print 'real value', Y[10]


