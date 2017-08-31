from CVmodelcomparison_qsubjob import *


def submit_jobs(datanum, modelnums=[0,1,2,3], nforecasts=1, minN_2_fit=20):
    '''
    Submit a new job for each train/test split of the RV time-series and each model. 
    E.g for 179 splits and 4 models, 716 jobs will be submitted.
    '''
    # check
    modelnums = np.ascontiguousarray(modelnums)
    assert not np.any(modelnums < 0)
    assert not np.any(modelnums > 3)
    
    # Define the sizes of the training sets
    t,_,_ = get_dataset(datanum)
    T = np.arange(int(minN_2_fit), t.size-int(nforecasts))
    
    # Run CV on each split, each forecast, and each model
    for m in modelnums:
        for i in range(nforecasts):
            for j in range(T.size):

                # Get jobscript
                f = open('jobscript_template', 'r')
                g = f.read()
                f.close()
                
                g = g.replace('<<datanum>>','%i'%datanum)
                g = g.replace('<<modelnum>>','%i'%m)
                g = g.replace('<<ind>>','%i'%int(T[j]))
                g = g.replace('<<nforecasts>>','%i'%nforecasts)
		g = g.replace('<<minN_2_fit>>','%i'%minN_2_fit)
                
                # Submit job
                h = open('jobscript', 'w')
                h.write(g)
                h.close()

                # Run job if not already
                if run_this_job(datanum, m, int(T[j]), nforecasts, minN_2_fit):
                    os.system('qsub jobscript')
                
		os.system('rm jobscript')
                #os.system('cat jobscript')


def run_this_job(datanum, modelnum, ind, nforecasts, minN_2_fit):
    folder = 'RVdata%.4d'%datanum
    outsuffix = 'qsubtest_modelnum%i_Ntrain%.3d_nforecasts%i_minN2fit%i'%(modelnum,
                                                                          ind,
                                                                          nforecasts,
                                                                          minN_2_fit)
    return not os.path.isfile('results/%s/%s'%(folder, outsuffix))
                

if __name__ == '__main__':
    datanum, modelnums = 1, [0,1,2,3]
    submit_jobs(datanum, modelnums)
