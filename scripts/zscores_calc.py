import glob # to search files using paths
import argparse  # to give an argument when executing the script

import re
import tables

import numpy as np
import statsmodels.api as sm

# INSERTING THE PATH OF THE DATACHECK FILE TO WORK WITH---------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="This is my code")

parser.add_argument('-f', '--input_files', dest='input_files', required=True,
                    type=str, help='Datacheck files to be processed')

parser.add_argument(
    "-n", "--max-files",
    type=int,
    default=None,
    help="Maximum number of files to process"
)

# input should be of the form: python my_script.py -f "/path/*.h5" -n 10


# ROBUST FIT FUNCTION 
def robust_fitt(x, y, z_thresh=3.5):
    X = x.astype(float).reshape(-1, 1)
    Xc = sm.add_constant(X)
    fit = np.zeros_like(y, dtype=float)
    good_points = []
    valid = []  # True/False per column
    good_masks = [] #to get the positions of the good points

    for p in range(y.shape[1]):
        yp = y[:, p]
        model = sm.OLS(yp, Xc).fit()
        residual = np.abs(yp - model.predict(Xc))
        mad = np.median(np.abs(residual - np.median(residual)))
        mad = max(mad, 1e-12)  # avoid division by zero
        modified_z = 0.6745 * residual / mad
        good = modified_z <= z_thresh
        good_masks.append(good)
        
        if good.sum() >= 0.6*len(x):  # if we take only 2 points, the line of the fit will cross them exactly, giving sigma=0 (both residuals=0). let's ask for 60% of the points to be selected
            good_points.append((x[good], yp[good]))
            
            model = sm.OLS(yp[good], Xc[good]).fit()
            fit[:, p] = model.predict(Xc)
            valid.append(True)
            
        else:
            good_points.append((None, None))
            fit[:, p] = np.nan
            valid.append(False)
        

    return {"fit": fit, "good_points": good_points, "good_mask": good_masks, "valid": valid}

def main():
    args = parser.parse_args()  #read arguments

    file_list = sorted(glob.glob(args.input_files)) #orders the fie
    
    if args.max_files is not None:
        file_list = file_list[:args.max_files]
    
    z_scores_h5 = tables.open_file(r"zscores.h5", mode="w")  # create our file
    
    # Variables for datachecks for which our model is valid (the fitting is possible), 
    class CheckResult(tables.IsDescription):
        z_score_1 = tables.Float64Col()
        z_score_2 = tables.Float64Col()
        run_id = tables.Int32Col()      
        subrun_index = tables.Int32Col() 

    t = z_scores_h5.create_table("/", "zscores", CheckResult, "zscores filter 1 and 2")

    # Variables for datachecks for which our model is NOT valid (could not do the fitting). Note: incorporate this in filter_results notebook

    class InvalidRuns(tables.IsDescription):
        run_id = tables.Int32Col()  
        filter_failed = tables.Int32Col()  

    t_invalid = z_scores_h5.create_table("/", "invalid_runs", InvalidRuns, "Runs where the robust fit could not be computed")

    # lists to take note of the runs for which our model is not valid     
    disfunctional_1 = []
    disfunctional_2 = []


    for file in file_list:
        a = tables.open_file(file)
        run = int(re.search(r"Run(\d+)", a.filename).group(1)) #extract the run number from the filename, gropu(0) takes 'Run24704' and group(1) takes the thing in brackets '24704'

# FILTERS TO DETECT ANOMALIES-----------------------------------------------------------------------------------------------------------------------------------        
        # USEFUL VARIABLES
        subruns = (a.root.dl1datacheck.cosmics.col('subrun_index')) #0,1,...,57 number of subruns in this run
        n_subruns = len(subruns)
        time = a.root.dl1datacheck.cosmics.col('elapsed_time')
        
        # if one of the filters work, the run will be stored in zscores.h5 and the other filter's variable will be replaced by nan
        # FILTER 1: EVENT RATE PER SUBRUN----------------------------------------------------------------------------------------------------
        num_events = a.root.dl1datacheck.cosmics.col('num_events') # events in each subrun
        event_rate = num_events / time
        event_rate_sigma = np.sqrt(np.maximum(num_events, 1.0)) / time  
        
        # Robust fit
        event_rate_robfit = robust_fitt(subruns, event_rate.reshape(-1, 1))

        # check that this fit is valid for this data
        if not event_rate_robfit["valid"][0]:
            disfunctional_1.append(run)
            row = t_invalid.row
            row["run_id"] = run
            row["filter_failed"] = 1
            row.append()   

            erate_z_score = np.full(n_subruns, np.nan)
        else:
            event_rate_fit = event_rate_robfit["fit"].flatten() # so to not work with ( , ) but only ()
            erate_z_score = np.abs(event_rate - event_rate_fit) / event_rate_sigma
        
        
        # FILTER 2: EVENT RATE VS INTENSITY----------------------------------------------------------------------------------------------------
        hist_intensity = a.root.dl1datacheck.cosmics.col('hist_intensity')
        
        cdf = np.cumsum(hist_intensity / np.maximum(hist_intensity.sum(axis=1, keepdims=True),1), axis=1) #we need to normalize the histogram. 1 cdf per subrun
        D = np.zeros(n_subruns)
        
        for s in range(0, n_subruns-1):
            D[s] = np.max(np.abs(cdf[s] - cdf[s+1])) #we compare the CDF in each subrun 's' with its neighbouring one 's+1'
        
        #Do a robust fit of D to select the good points
        D_robust_fit = robust_fitt(subruns, D.reshape(-1, 1))
        
        if not D_robust_fit["valid"][0]:
            disfunctional_2.append(run)
            row = t_invalid.row
            row["run_id"] = run
            row["filter_failed"] = 2
            row.append()
          
            deviations_2 = np.full(n_subruns, np.nan)
            #print(f"Filter 2 skipped for Run {run}, not enough good points for a reliable linear robust fit.")
            
        else:
            # This is not a Poisson process therefore we cannot treat it as such
            D_good = D_robust_fit["good_points"][0][1]  # y-selected values for this column
            D_fit = D_robust_fit["fit"].flatten()       
      
            good_mask_2 = D_robust_fit["good_mask"][0]               
            D_fit_good = D_fit[good_mask_2]                # take the fit only at the 'good points' to calculate the sigmas
            
            sigma_2 = (np.std(D_good - D_fit_good))
            deviations_2 = np.abs(D - D_fit) / sigma_2    # compare all of the values of D to all of the values of the fit
        


        if not event_rate_robfit["valid"][0] and not D_robust_fit["valid"][0]:
            pass     #we do not write the run in zscores.h5 if none of the filters work
        else:
                row_good = t.row
                for sr, z1, z2 in zip(subruns, erate_z_score, deviations_2):
                    row_good["run_id"] = run
                    row_good["subrun_index"] = sr
                    row_good["z_score_1"] = z1
                    row_good["z_score_2"] = z2
                    row_good.append()

        a.close()
        
    print("Filter 1 could not be applied to ", len(disfunctional_1), " Runs: ", disfunctional_1)
    print("Filter 2 could not be applied to ", len(disfunctional_2), " Runs: ", disfunctional_2)

    t.flush()
    t_invalid.flush()
 
    z_scores_h5.close()

if __name__ == '__main__':
    main()

