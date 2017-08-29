import numpy as np
import pylab as plt
import scipy.optimize as op
import sys
import os
import george
from george import kernels
import emcee
import corner
import cPickle as pickle
from PyAstronomy.pyasl import foldAt
import glob
import time
import rvmodel as rvm
