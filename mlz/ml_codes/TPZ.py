"""
.. module:: TPZ
.. moduleauthor:: Matias Carrasco Kind
"""
__author__ = 'Matias Carrasco Kind'
from numpy import *
import random as rn
import os


def split_point(data1, yvals, minleaf):
    """
    Computes the best point where the data should be split in two
    given a dimension, returns the variance and the splitting value
    """
    serr = 1.0e30
    sort_d = argsort(data1)
    data1 = data1[sort_d]
    yvals = yvals[sort_d]
    Ntot = len(data1)
    u = unique(data1)
    if len(u) == 1:
        return serr, u[0]
    if len(u) <= minleaf:
        for ju in xrange(len(u) - 1):
            win = where(data1 == u[ju])[0][-1]
            nin = win + 1
            nsplit = nin
            stemp = var(yvals[0:nin]) * nin + var(yvals[nin:]) * (Ntot - nin)
            if stemp < serr:
                serr = stemp
                nsplit = nin
        bbb = 0.5 * (data1[nsplit - 1] + data1[nsplit])
        return serr, bbb
    nin = 1
    nsplit = nin
    ntimes = Ntot - 2 * minleaf + 1
    for i in xrange(ntimes):
        stemp = var(yvals[0:nin]) * nin + var(yvals[nin:]) * (Ntot - nin)
        if stemp < serr:
            serr = stemp
            nsplit = nin
        nin += 1
    bbb = 0.5 * (data1[nsplit - 1] + data1[nsplit])
    return serr, bbb


def best_split(data_split, y_split, minleaf):
    """
    Computes the best splitting point among all available dimensions
    """
    Ndim = shape(data_split)[1]
    Sall = zeros(Ndim)
    vall = zeros(Ndim)
    for i in xrange(Ndim):
        values = data_split[:, i]
        Ssplit, vsplit = split_point(values, y_split, minleaf)
        Sall[i] = Ssplit
        vall[i] = vsplit
    minS = argmin(Sall)
    maxS = argmax(Sall)
    if maxS == minS: return -1, 0., 0.
    return minS, vall[minS], Sall[minS]


def I_d(y_vals, impurity, nclass):
    """
    Impurity function
    """
    tot = len(y_vals) * 1.
    if tot == 0.:
        print 'ERROR'
        return -1.
    Idd = 0.
    if impurity == 'gini': Idd = 1.
    if impurity == 'classE':
        fn0 = -0.1
    for n in nclass:
        fn = 1. * sum(y_vals == n) / tot
        if impurity == 'entropy':
            if fn > 0.: Idd += -fn * log2(fn)
        if impurity == 'gini': Idd -= fn * fn
        if impurity == 'classE':
            mfn = max(fn0, fn)
            fn0 = fn
    if impurity == 'classE': Idd = 1. - mfn
    return Idd


def gain(vals_x, vals_y, minleaf, impurity, nclass):
    """
    Information gain
    """
    tot = len(vals_x)
    IG = I_d(vals_y, impurity, nclass)
    max_IG = -inf
    sort_d = argsort(vals_x)
    vals_x = vals_x[sort_d]
    vals_y = vals_y[sort_d]
    u = unique(vals_x)
    if len(u) == 1: return 0., u[0]
    if len(u) <= minleaf:
        for ju in xrange(len(u) - 1):
            win = where(vals_x == u[ju])[0][-1]
            nin = win + 1
            nsplit = nin
            nout = 1. * (tot - nin)
            IGtemp = IG - 1. * nin / (1. * tot) * I_d(vals_y[0:nin], impurity, nclass) - 1. * nout / (1. * tot) * I_d(
                vals_y[nin:], impurity, nclass)
            if IGtemp > max_IG:
                max_IG = IGtemp
                nsplit = nin
        bbb = 0.5 * (vals_x[nsplit - 1] + vals_x[nsplit])
        return max_IG, bbb
    nin = 1
    nsplit = nin
    ntimes = int(tot - 2 * minleaf + 1)
    for i in xrange(ntimes):
        nout = 1. * (tot - nin)
        IGtemp = IG - 1. * nin / (1. * tot) * I_d(vals_y[0:nin], impurity, nclass) - 1. * nout / (1. * tot) * I_d(
            vals_y[nin:], impurity, nclass)
        if IGtemp > max_IG:
            max_IG = IGtemp
            nsplit = nin
        nin += 1
    bbb = 0.5 * (vals_x[nsplit - 1] + vals_x[nsplit])
    return max_IG, bbb


def best_split_class(data_split, y_split, minleaf, impurity, nclass):
    Ndim = shape(data_split)[1]
    IGall = zeros(Ndim)
    vall = zeros(Ndim)
    for i in xrange(Ndim):
        values = data_split[:, i]
        IGsplit, vsplit = gain(values, y_split, minleaf, impurity, nclass)
        IGall[i] = IGsplit
        vall[i] = vsplit
    maxIG = argmax(IGall)
    minIG = argmin(IGall)
    if maxIG == minIG:
        return -1, 0., 0.
    return maxIG, vall[maxIG], IGall[maxIG]

############################################################


class InsertNode():
    """
    Add a node to a tree during the growing process
    """

    def __init__(self, sdim, spt, nl, nr, depth, left, right):
        self.dim = sdim
        self.point = spt
        self.depth = depth
        self.nl = nl
        self.nr = nr
        self.left = left
        self.right = right
        self.is_L_leaf = False
        self.is_R_leaf = False
        if type(self.left) == type(ones(1)): self.is_L_leaf = True
        if type(self.right) == type(ones(1)): self.is_R_leaf = True


class Ctree():
    """
    Creates a  classification tree class instance

    :param X: Preprocessed attributes array (*all* columns are considered)
    :type X: float or int array, 1 row per object
    :param Y: Attribute to be predicted
    :type Y: int array
    :param minleaf: Minimum number of objects on terminal leaf
    :type minleaf: int, def = 4
    :param forest: Random forest key
    :type forest:   str, 'yes'/'no'
    :param mstar: Number of random subsample of attributes if forest is used
    :type mstar: int
    :param impurity: 'entropy'/'gini'/'classE' to compute information gain
    :param nclass: classes array (labels)
    :type nclass: int array
    :param dict_dim: dictionary with attributes names
    :type dict_dim: dict
    """

    def __init__(self, X, Y, minleaf=4, forest='yes', mstar=2, dict_dim='', impurity='entropy',
                 nclass=arange(2, dtype='int')):
        self.dict_dim = dict_dim
        self.nclass = nclass
        self.xdim = shape(X)[1]
        self.nobj = len(X)

        def build(XD, YD, depth):
            ND = len(XD)
            if ND <= minleaf:
                return YD
            self.depth = depth
            all_D = arange(shape(XD)[1])
            if forest == 'no':
                myD = all_D
            if forest == 'yes':
                myD = rn.sample(all_D, mstar)
            if len(unique(YD)) == 1: return YD
            #if len(unique(XD[:,0]))==1: return YD
            td, sp, tvar = best_split_class(XD[:, myD], YD, minleaf, impurity, nclass)
            if td == -1: return YD
            sd = myD[td]
            S = where(XD[:, sd] <= sp, 1, 0)
            wL = where(S == 1.)[0]
            wR = where(S == 0.)[0]
            NL = shape(wL)[0]
            NR = shape(wR)[0]
            node = InsertNode(sd, sp, NL, NR, depth, left=build(XD[wL], YD[wL], depth + 1),
                              right=build(XD[wR], YD[wR], depth + 1))
            return node

        self.root = build(X, Y, 0)

    def leaves(self):
        """
        Same as :func:`Rtree.leaves`
        """
        bt = Ls(self.root, B=[], O=[])
        return bt

    def leaves_dim(self):
        """
        Same as :func:`Rtree.leaves_dim`
        """
        bt = Ls_dim(self.root, B=[], O=[])
        return bt

    def get_branch(self, line):
        """
        Same as :func:`Rtree.get_branch`
        """
        return search_B(line, self.root, SB=[])

    def get_vals(self, line):
        """
        Same as :func:`Rtree.get_vals`
        """
        out = search(line, self.root)
        if len(out) >= 1: return out
        return array([-1.])

    def print_branch(self, branch):
        """
        Same as :func:`Rtree.print_branch`
        """
        return Pb(self.root, branch)

    def save_tree(self, itn=-1, fileout='TPZ', path=''):
        """
        Same as :func:`Rtree.save_tree`
        """
        if path == '':
            path = os.getcwd() + '/'
        if not os.path.exists(path): os.system('mkdir -p ' + path)
        if itn >= 0:
            ff = '_%04d' % itn
            fileout += ff
        save(path + fileout, self)

    def plot_tree(self, itn=-1, fileout='TPZ', path='', save_png='no'):
        """
        Same as :func:`Rtree.plot_tree`
        """
        import matplotlib.pyplot as plt

        if path == '':
            path = os.getcwd() + '/'
        if not os.path.exists(path): os.system('mkdir -p ' + path)
        if itn >= 0:
            ff = '_%04d' % itn
            fileout += ff
        fdot3 = open(fileout + '.dot', 'w')
        fdot3.write('digraph \" TPZ Tree \" { \n')
        fdot3.write('''size="15,80" ;\n ''')
        fdot3.write('''ratio=auto; \n ''')
        fdot3.write('''overlap=false; \n ''')
        fdot3.write('''spline=true; \n ''')
        fdot3.write('''sep=0.02; \n ''')
        fdot3.write('''bgcolor=white; \n ''')
        fdot3.write('''node [shape=circle, style=filled]; \n ''')
        fdot3.write('''edge [arrowhead=none, color=black, penwidth=0.4]; \n''')
        colors = array(
            ['purple', 'blue', 'green', 'magenta', 'brown', 'orange', 'yellow', 'aquamarine', 'cyan', 'lemonchiffon'])
        shapes = array(['circle', 'square', 'triangle', 'polygon', 'diamond', 'star'])
        colors_class = array(['black', 'red', 'gray', 'black', 'red', 'gray'])
        Leaf = self.leaves()
        Leaf_dim = array(self.leaves_dim())
        node_dict = {}
        for il in xrange(len(Leaf)):
            Lnum = branch2num(Leaf[il])
            Lclass = int(round(mean(self.print_branch(Leaf[il]))))
            ldim = Leaf_dim[il]
            for jl in xrange(len(Lnum) - 1):
                n1 = Lnum[jl]
                if not node_dict.has_key(n1):
                    n2 = Lnum[jl + 1]
                    node_dict[n1] = [n2]
                    node1 = 'node_' + str(n1)
                    node2 = 'node_' + str(n2)
                    if jl == len(Lnum) - 2: node2 = 'Leaf_' + str(n2)
                    line = node1 + ' -> ' + node2 + ';\n'
                    fdot3.write(line)
                    line = node1 + '''[label="", height=0.3, fillcolor=''' + colors[ldim[jl]] + '] ; \n'
                    fdot3.write(line)
                    if jl == len(Lnum) - 2:
                        line = node2 + '''[shape=''' + shapes[
                            Lclass] + ''',label="", height=0.1, fillcolor=''' + colors_class[Lclass] + '''] ; \n'''
                        fdot3.write(line)
                else:
                    n2 = Lnum[jl + 1]
                    if not n2 in node_dict[n1]:
                        node1 = 'node_' + str(n1)
                        node_dict[n1].append(n2)
                        node2 = 'node_' + str(n2)
                        if jl == len(Lnum) - 2: node2 = 'Leaf_' + str(n2)
                        line = node1 + ' -> ' + node2 + ';\n'
                        fdot3.write(line)
                        if jl == len(Lnum) - 2:
                            line = node2 + '''[shape=''' + shapes[
                                Lclass] + ''',label="", height=0.1, fillcolor=''' + colors_class[Lclass] + '''] ; \n'''
                            fdot3.write(line)
        fdot3.write('}')
        fdot3.close()
        os.system('neato -Tpng ' + fileout + '.dot > ' + fileout + '_neato.png')
        os.system('dot -Tpng ' + fileout + '.dot > ' + fileout + '_dot.png')
        Adot = plt.imread(fileout + '_dot.png')
        Aneato = plt.imread(fileout + '_neato.png')
        plt.figure(figsize=(14, 8))
        plt.imshow(Adot)
        plt.axis('off')
        plt.figure(figsize=(12, 12))
        plt.imshow(Aneato)
        plt.axis('off')
        plt.figure(figsize=(8, 2.), facecolor='white')
        if self.dict_dim == '' or self.dict_dim == 'all':
            for i in xrange(self.xdim):
                plt.scatter(i, 1, s=250, c=colors[i])
                plt.text(i, 0.5, str(i), ha='center', rotation='40')
        else:
            for ik in self.dict_dim.keys():
                plt.scatter(self.dict_dim[ik]['ind'], 1, s=250, c=colors[self.dict_dim[ik]['ind']])
                plt.text(self.dict_dim[ik]['ind'], 0.5, ik, ha='center', rotation='40')
        plt.xlim(-1, self.xdim + 1)
        plt.ylim(0, 2)
        plt.axis('off')
        plt.show()
        if save_png == 'no':
            os.remove(fileout + '_neato.png')
            os.remove(fileout + '_dot.png')
            os.remove(fileout + '.dot')


class Rtree():
    """
    Creates a  regression tree class instance

    .. todo::
        Add weights to objects

    :param X: Preprocessed attributes array (*all* columns are considered)
    :type X: float or int array, 1 row per object
    :param float Y: Attribute to be predicted
    :param minleaf: Minimum number of objects on terminal leaf
    :type minleaf: int, def = 4
    :param forest: Random forest key
    :type forest:   str, 'yes'/'no'
    :param mstar: Number of random subsample of attributes if forest is used
    :type mstar: int
    :param dict_dim: dictionary with attributes names
    :type dict_dim: dict

    .. literalinclude:: testf.py
        :linenos:
    """

    def __init__(self, X, Y, minleaf=4, forest='yes', mstar=2, dict_dim=''):
        self.dict_dim = dict_dim
        self.xdim = shape(X)[1]
        self.nobj = len(X)

        def build(XD, YD, depth):
            ND = len(XD)
            if ND <= minleaf:
                return YD
            self.depth = depth
            all_D = arange(shape(XD)[1])
            if forest == 'no':
                myD = all_D
            if forest == 'yes':
                myD = rn.sample(all_D, mstar)
            if len(unique(YD)) == 1: return YD
            td, sp, tvar = best_split(XD[:, myD], YD, minleaf)
            if td == -1: return YD
            sd = myD[td]
            S = where(XD[:, sd] <= sp, 1, 0)
            wL = where(S == 1.)[0]
            wR = where(S == 0.)[0]
            NL = shape(wL)[0]
            NR = shape(wR)[0]
            node = InsertNode(sd, sp, NL, NR, depth, left=build(XD[wL], YD[wL], depth + 1),
                              right=build(XD[wR], YD[wR], depth + 1))
            return node

        self.root = build(X, Y, 0)

    def leaves(self):
        """
        Return an array with all branches in string format
        ex: ['L','R','L'] is a branch of depth 3 where L and R are the left or right
        branches

        :returns: str -- Array of all branches in the tree
        """
        bt = Ls(self.root, B=[], O=[])
        return bt

    def leaves_dim(self):
        """
        Returns an array of the used dimensions for all the the nodes on all the branches

        :return: int -- Array of all the dimensions for each node on each branch
        """
        bt = Ls_dim(self.root, B=[], O=[])
        return bt

    def get_branch(self, line):
        """
        Get the branch in string format given a line search, where the line
        is a vector of attributes per individual object

        :param float line: input data line to look in the tree, same dimensions as input X
        :returns: str -- branch array in string format, ex., ['L','L','R']
        """
        return search_B(line, self.root, SB=[])

    def get_vals(self, line):
        """
        Get the predictions  given a line search, where the line
        is a vector of attributes per individual object

        :param float line: input data line to look in the tree, same dimensions as input X
        :returns: float -- array with the leaf content
        """
        out = search(line, self.root)
        if len(out) >= 1:
            return out
        return array([-1.])

    def print_branch(self, branch):
        """
        Returns the content of a leaf on a branch (given in string format)
        """
        return Pb(self.root, branch)

    def save_tree(self, itn=-1, fileout='TPZ', path=''):
        """
        Saves the tree

        :param int itn: Number of tree to be included on path, use -1 to ignore this number
        :param str fileout: Name of output file
        :param str path: path for the output file
        """
        if path == '':
            path = os.getcwd() + '/'
        if not os.path.exists(path): os.system('mkdir -p ' + path)
        if itn >= 0:
            ff = '_%04d' % itn
            fileout += ff
        save(path + fileout, self)

    def plot_tree(self, itn=-1, fileout='TPZ', path='', save_png='no'):
        """
        Plot a tree using dot (Graphviz)
        Saves it into a png file by default

        :param int itn: Number of tree to be included on path, use -1 to ignore this number
        :param str fileout: Name of file for the png files
        :param str path: path for the output files
        :arg str save_png: save png created by Graphviz ('yes'/'no')
        """
        import matplotlib.pyplot as plt

        if path == '':
            path = os.getcwd() + '/'
        if not os.path.exists(path): os.system('mkdir -p ' + path)
        if itn >= 0:
            ff = '_%04d' % itn
            fileout += ff
        fdot3 = open(fileout + '.dot', 'w')
        fdot3.write('digraph \" TPZ Tree \" { \n')
        fdot3.write('''size="15,80" ;\n ''')
        fdot3.write('''ratio=auto; \n ''')
        fdot3.write('''overlap=false; \n ''')
        fdot3.write('''spline=true; \n ''')
        fdot3.write('''sep=0.02; \n ''')
        fdot3.write('''bgcolor=white; \n ''')
        fdot3.write('''node [shape=circle, style=filled]; \n ''')
        fdot3.write('''edge [arrowhead=none, color=black, penwidth=0.4]; \n''')
        colors = array(
            ['purple', 'blue', 'green', 'magenta', 'brown', 'orange', 'yellow', 'aquamarine', 'cyan', 'lemonchiffon'])
        Leaf = self.leaves()
        Leaf_dim = array(self.leaves_dim())
        node_dict = {}
        for il in xrange(len(Leaf)):
            Lnum = branch2num(Leaf[il])
            ldim = Leaf_dim[il]
            for jl in xrange(len(Lnum) - 1):
                n1 = Lnum[jl]
                if not node_dict.has_key(n1):
                    n2 = Lnum[jl + 1]
                    node_dict[n1] = [n2]
                    node1 = 'node_' + str(n1)
                    node2 = 'node_' + str(n2)
                    if jl == len(Lnum) - 2: node2 = 'Leaf_' + str(n2)
                    line = node1 + ' -> ' + node2 + ';\n'
                    fdot3.write(line)
                    line = node1 + '''[label="", height=0.3, fillcolor=''' + colors[ldim[jl]] + '] ; \n'
                    fdot3.write(line)
                    if jl == len(Lnum) - 2:
                        line = node2 + '''[label="", height=0.1, fillcolor=black] ; \n'''
                        fdot3.write(line)
                else:
                    n2 = Lnum[jl + 1]
                    if not n2 in node_dict[n1]:
                        node1 = 'node_' + str(n1)
                        node_dict[n1].append(n2)
                        node2 = 'node_' + str(n2)
                        if jl == len(Lnum) - 2: node2 = 'Leaf_' + str(n2)
                        line = node1 + ' -> ' + node2 + ';\n'
                        fdot3.write(line)
                        if jl == len(Lnum) - 2:
                            line = node2 + '''[label="", height=0.1, fillcolor=black] ; \n'''
                            fdot3.write(line)
        fdot3.write('}')
        fdot3.close()
        os.system('neato -Tpng ' + fileout + '.dot > ' + fileout + '_neato.png')
        os.system('dot -Tpng ' + fileout + '.dot > ' + fileout + '_dot.png')
        Adot = plt.imread(fileout + '_dot.png')
        Aneato = plt.imread(fileout + '_neato.png')
        plt.figure(figsize=(14, 8))
        plt.imshow(Adot)
        plt.axis('off')
        plt.figure(figsize=(12, 12))
        plt.imshow(Aneato)
        plt.axis('off')
        plt.figure(figsize=(8, 2.), facecolor='white')
        if self.dict_dim == '' or self.dict_dim == 'all':
            for i in xrange(self.xdim):
                plt.scatter(i, 1, s=250, c=colors[i])
                plt.text(i, 0.5, str(i), ha='center', rotation='40')
        else:
            for ik in self.dict_dim.keys():
                plt.scatter(self.dict_dim[ik]['ind'], 1, s=250, c=colors[self.dict_dim[ik]['ind']])
                plt.text(self.dict_dim[ik]['ind'], 0.5, ik, ha='center', rotation='40')
        plt.xlim(-1, self.xdim + 1)
        plt.ylim(0, 2)
        plt.axis('off')
        plt.show()
        if save_png == 'no':
            os.remove(fileout + '_neato.png')
            os.remove(fileout + '_dot.png')
            os.remove(fileout + '.dot')


# EXTRA FUNCTIONS USED IN CLASS
def search_B(line, node, SB=[]):
    sd = node.dim
    pt = node.point
    if line[sd] < pt:
        SB.append('L')
        if node.is_L_leaf:
            return SB
        else:
            return search_B(line, node.left, SB)
    else:
        SB.append('R')
        if node.is_R_leaf:
            return SB
        else:
            return search_B(line, node.right, SB)


def search(line, node):
    sd = node.dim
    pt = node.point
    if line[sd] < pt:
        if node.is_L_leaf:
            return node.left
        else:
            return search(line, node.left)
    else:
        if node.is_R_leaf:
            return node.right
        else:
            return search(line, node.right)


def Ls(node, B=[], O=[]):
    if node.is_L_leaf:
        B.append('L')
        O.append(B)
        B = B[:-1]
    else:
        B.append('L')
        O = Ls(node.left, B)
        B = B[:node.depth]
    if node.is_R_leaf:
        B.append('R')
        O.append(B)
        B = B[:-2]
    else:
        B.append('R')
        O = Ls(node.right, B)
        B = B[:node.depth]
    return O


def Ls_dim(node, B=[], O=[]):
    if node.is_L_leaf:
        B.append(node.dim)
        O.append(B)
        B = B[:-1]
    else:
        B.append(node.dim)
        O = Ls_dim(node.left, B)
        B = B[:node.depth]
    if node.is_R_leaf:
        B.append(node.dim)
        O.append(B)
        B = B[:-2]
    else:
        B.append(node.dim)
        O = Ls_dim(node.right, B)
        B = B[:node.depth]
    return O


def Pb(node, branch):
    temp = node
    for b in branch:
        if b == 'L':
            temp = temp.left
        if b == 'R':
            temp = temp.right
    return temp


def branch2num(branch):
    num = [0]
    for b in branch:
        if b == 'L':
            num.append(num[-1] * 2 + 1)
        if b == 'R':
            num.append(num[-1] * 2 + 2)
    return num
     

