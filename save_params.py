from savepickle import *
from run_MCMC import get_results_kernel


def eo2hk(e,o):
    return e*np.cos(o), e*np.sin(o)


def save_params(datanum, modelnum):
    self = loadpickle('results/RVdata%.4d/qsub_datanum%i_modelnum%i'%(datanum,
                                                                      datanum,
                                                                      modelnum))

    thetas = get_results_kernel(self.thetas, pltt=0)[:,0]
    theta_medians = get_results_kernel(self.theta_medians, pltt=0)[:,0]

    paramlabels = ['P','K','h','k']*modelnum
    paramlabels.append('C')
    paramlabels.append('J')
    order = np.append([2,4,5,6,7,9,10,11,12,14,15,16][:modelnum*4], [1,0])
    order = order.astype(int)
    thetas = thetas[order]
    theta_medians = theta_medians[order]
    
    N, g = thetas.size, ''
    for i in range(N):
        g += '1.60E+07,%s,%i,'%(paramlabels[i], modelnum)
        
        if i == N-2:
            g += '0,'

        elif i == N-1:
            g += '0,'
            
        else:
            if i < 4:
                planetnum = 1
            elif 4 <= i < 8:
                planetnum = 2
            else:
                planetnum = 3
            g+= '%i,'%planetnum

        if paramlabels[i] == 'h': 
            h, k = eo2hk(thetas[i], thetas[i+1])
            g += '%.8e,'%h
            hm, km = eo2hk(theta_medians[i], theta_medians[i+1])
            g += '%.8e,,,,'%hm
        elif paramlabels[i] == 'k':
            g += '%.8e,'%k
            g += '%.8e,,,,'%km
        else:
            g += '%.8e,'%thetas[i]
            g += '%.8e,,,,'%theta_medians[i]
            
        g += '\n'

    h = open('Cloutier/TimeSeriesCV/params_%i_%.4d.txt'%(modelnum, datanum),'w')
    h.write(g)
    h.close()
