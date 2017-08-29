from CVmodelcomparison import *


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
		g = g.replace('<<min_2_fit>>','%i'%min_2_fit)
                
                # Submit job
                h = open('jobscript', 'w')
                h.write(g)
                h.close()
                
                #os.system('qsub jobscript')
                #os.sytem('rm jobscript')
                os.system('cat jobscript')
                
                #ind = int(T[j])
                #lnlike, success, theta0, thetaopt, = compute_modelposterior_CV(datanum, m, ind, nforecasts)
                
