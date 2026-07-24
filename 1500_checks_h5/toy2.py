import glob # to search files using paths
import argparse  # to give an argument when executing the script

import re
import tables

import numpy as np
import statsmodels.api as sm


# INSERTING THE PATH OF THE DATACHECK FILE TO WORK WITH---------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="This is my code") # parser is the object that will read the arguments, description will appear when python prueba.py --help

parser.add_argument('-f', '--input_files', dest='input_files', required=True,
                    type=str, help='Datacheck files to be processed')
# defining an argument (the 'input') 
# -f short way to call the file
# --input files long way
# required=True makes mandetory to provide an argument

parser.add_argument(      # batch = set of files 
    "--batch-size",
    type=int,
    default=None,
    help="Number of files per batch"
)

parser.add_argument(     # starting point if the set of files
    "--batch",
    type=int,
    default=0,
    help="Batch number (starting from 0)"
)

# ROBUST FIT FUNCTION 
def robust_fitt(x, y, z_thresh=3.5):
    X = x.astype(float).reshape(-1, 1)
    Xc = sm.add_constant(X)
    fit = np.zeros_like(y, dtype=float)
    slope = np.zeros(y.shape[1])
    u_slope = np.zeros(y.shape[1])
    intercept = np.zeros(y.shape[1])
    u_intercept = np.zeros(y.shape[1])
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
        
        if good.sum() >= 0.6*len(x):  # if we take only 2 points, the line of the fit will cross them exactly, giving sigma=0 (both residuals=0)
            good_points.append((x[good], yp[good]))
            
            model = sm.OLS(yp[good], Xc[good]).fit()
            fit[:, p] = model.predict(Xc)
            intercept[p] = model.params[0]
            slope[p] = model.params[1]
            u_intercept[p] = model.bse[0]
            u_slope[p] = model.bse[1]
            
            valid.append(True)
            
        else:
            good_points.append((None, None))
            fit[:, p] = np.nan
            intercept[p] = np.nan
            slope[p] = np.nan
            u_intercept[p] = np.nan
            u_slope[p] = np.nan
            valid.append(False)
        

    return {"fit": fit, "slope": slope, "u_slope": u_slope,
            "intercept": intercept, "u_intercept": u_intercept,
            "good_points": good_points, "good_mask": good_masks, "valid": valid}

"""
fit          :  (len(x), y.shape[1])
slope        :  (y.shape[1],)
u_slope      :  (y.shape[1],)
intercept    :  (y.shape[1],)
u_intercept  :  (y.shape[1],)
good_points  : list of (x_good, y_good) 
good_mask    : list 
valid        : list length y.shape[1]
"""


# FILTERS FUNCTIONS
"""
sigma : uncertainty associated with the input data (Poisson's, stdev,...)
z_scores : # of sigmas by which a point in the data deviates from the linear fit
sigma_cutoff : # of sigmas by which we determine a zscore as anomalous

"""
def f1(subruns, event_rate, event_rate_sigma, sigma_cutoff):

    event_rate_robfit = robust_fitt(subruns, event_rate.reshape(-1, 1))

    if not event_rate_robfit["valid"][0]:
        return {
            "valid": False,
            "checks": np.array([], dtype=np.int64),
            "z_score": np.array([], dtype=np.float64),
            "slope": np.nan,
            "u_slope": np.nan,
            "intercept": np.nan,
            "u_intercept": np.nan,
        }

    else:
        event_rate_fit = event_rate_robfit["fit"].flatten()

        erate_z_score = np.abs(event_rate - event_rate_fit) / event_rate_sigma

        mask = erate_z_score > sigma_cutoff

        return {
            "valid": True,
            "checks": subruns[mask],
            "z_score": erate_z_score[mask],
            "slope": event_rate_robfit["slope"][0],
            "u_slope": event_rate_robfit["u_slope"][0],
            "intercept": event_rate_robfit["intercept"][0],
            "u_intercept": event_rate_robfit["u_intercept"][0],
        }
    


def f2(subruns, cdf, sigma_cutoff):
    """
    if cdf.shape[1] == 0:
    return {
        "valid": False,
        "checks": np.array([], dtype=np.int64),
        "z_score": np.array([], dtype=np.float64),
        "slope": np.nan,
        "u_slope": np.nan,
        "intercept": np.nan,
        "u_intercept": np.nan,
    }
    """
    n_subruns = len(subruns)
    D = np.zeros(n_subruns)

    for s in range(n_subruns - 1):
        D[s] = np.max(np.abs(cdf[s] - cdf[s + 1]))

    # Robust fit
    D_robust_fit = robust_fitt(subruns, D.reshape(-1, 1))

    if not D_robust_fit["valid"][0]:
        return {
            "valid": False,
            "checks": np.array([], dtype=np.int64),
            "z_score": np.array([], dtype=np.float64),
            "slope": np.nan,
            "u_slope": np.nan,
            "intercept": np.nan,
            "u_intercept": np.nan,
        }

    else:
        # This is not a Poisson process therefore we cannot treat it as such
        D_good = D_robust_fit["good_points"][0][1]
        D_fit = D_robust_fit["fit"].flatten()

        good_mask = D_robust_fit["good_mask"][0]
        D_fit_good = D_fit[good_mask]

        sigma = np.std(D_good - D_fit_good)

        deviations = np.abs(D - D_fit) / sigma

        mask = deviations > sigma_cutoff

        return {
            "valid": True,
            "checks": subruns[mask],
            "z_score": deviations[mask],
            "slope": D_robust_fit["slope"][0],
            "u_slope": D_robust_fit["u_slope"][0],
            "intercept": D_robust_fit["intercept"][0],
            "u_intercept": D_robust_fit["u_intercept"][0],
        }


def main():
    args = parser.parse_args()  #read arguments
    file_list = sorted(glob.glob(args.input_files))

    if args.batch_size is not None:
        start = args.batch * args.batch_size
        end = start + args.batch_size
        file_list = file_list[start:end]

    # lists to store the id of the runs for which our model is not valid
    disfunctional_1 = []
    disfunctional_2 = []

    invalid_runs = [np.int32(20908), np.int32(21187), np.int32(21188), np.int32(21387), np.int32(21467), np.int32(21474), np.int32(21509), np.int32(21778), np.int32(21818), np.int32(21877), np.int32(22259), np.int32(22685), np.int32(22737), np.int32(23045), np.int32(20735)]

    for file in file_list:
        a = tables.open_file(file)
        run = int(re.search(r"Run(\d+)", a.filename).group(1)) #extract the run number from the filename, gropu(0) takes 'Run24704' and group(1) takes the thing in brackets '24704'
        
        if run in invalid_runs:
                print(f"Skipping Run {run}: both filters invalid")
                a.close()
                continue

# FILTERS TO DETECT ANOMALIES-----------------------------------------------------------------------------------------------------------------------------------
        dictionary = {}
        # USEFUL VARIABLES
        subruns = a.root.dl1datacheck.cosmics.col('subrun_index')[:-1] #0,1,...,57 number of subruns in this run
        time = a.root.dl1datacheck.cosmics.col('elapsed_time')[:-1] 
        
        
        # FILTER 1: EVENT RATE PER SUBRUN----------------------------------------------------------------------------------------------------
        num_events = a.root.dl1datacheck.cosmics.col('num_events')[:-1]  # events in each subrun
        event_rate = num_events / time
        event_rate_sigma = np.sqrt(np.maximum(num_events, 1.0)) / time  

        sigma_cutoff_1 = 20 

        result_1 = f1(subruns, event_rate, event_rate_sigma, sigma_cutoff_1)

        if not result_1["valid"]:
            disfunctional_1.append(run)


        dictionary.update({
            "checks_1": result_1["checks"],
            "z_score_1": result_1["z_score"],
            "slope_1": result_1["slope"],
            "u_slope_1": result_1["u_slope"],
            "intercept_1": result_1["intercept"],
            "u_intercept_1": result_1["u_intercept"],
        })



        # FILTER 2: CDF OF THE INTENSITIES PER SUBRUN----------------------------------------------------------------------------------------------
        hist_intensity = a.root.dl1datacheck.cosmics.col('hist_intensity')[:-1] 
        intensity_bins = a.root.dl1datacheck.histogram_binning.col('hist_intensity')[0]
           
        # Event rate per intensity value in each subrun
        # Therefore, to perform a more sensitive analysis we'll implement the Kolmogorov-Smirnov statistic, because we want to compare the whole shape of our distribuition
        # with that end we implement the Kolmogorov-Smirnov parameter D = max |F(x) - Fn| which tells us how different are two cumulative distribuition functions
        # we will compare the CDF for each subrun with its neighbours, bc the satellite is expected to crossthe FoV within a fine period of time

        int_a = np.searchsorted(intensity_bins, 100, side="right") #intensities up to 100 p.e.
        int_b = np.searchsorted(intensity_bins, 1000, side="right") # intensities up to 1000 p.e.
        
        cdf_all = np.cumsum(hist_intensity / np.maximum(hist_intensity.sum(axis=1, keepdims=True),1), axis=1) #we need to normalize the histogram. 1 cdf per subrun

        cdf_a = np.cumsum(hist_intensity[:, :int_a] / np.maximum(hist_intensity[:, :int_a].sum(axis=1, keepdims=True), 1),axis=1)
        cdf_b = np.cumsum(hist_intensity[:, int_a:int_b] / np.maximum(hist_intensity[:, int_a:int_b].sum(axis=1, keepdims=True), 1),axis=1)
        cdf_c = np.cumsum(hist_intensity[:, int_b:] / np.maximum(hist_intensity[:, int_b:].sum(axis=1, keepdims=True), 1),axis=1)

        sigma_cutoff_2 = 10
        sigma_cutoff_2a = 6
        sigma_cutoff_2b = 10
        sigma_cutoff_2c = 10

        result_2 = f2(subruns, cdf_all, sigma_cutoff_2)
        result_2a = f2(subruns, cdf_a, sigma_cutoff_2a)
        result_2b = f2(subruns, cdf_b, sigma_cutoff_2b)
        result_2c = f2(subruns, cdf_c, sigma_cutoff_2c)

        if not result_2["valid"]:
            disfunctional_2.append((run, "all"))

        if not result_2a["valid"]:
            disfunctional_2.append((run, "a"))

        if not result_2b["valid"]:
            disfunctional_2.append((run, "b"))

        if not result_2c["valid"]:
            disfunctional_2.append((run, "c"))

        dictionary.update({
            "checks_2": result_2["checks"],
            "z_score_2": result_2["z_score"],
            "slope_2": result_2["slope"],
            "u_slope_2": result_2["u_slope"],
            "intercept_2": result_2["intercept"],
            "u_intercept_2": result_2["u_intercept"],

            "checks_2a": result_2a["checks"],
            "z_score_2a": result_2a["z_score"],
            "slope_2a": result_2a["slope"],
            "u_slope_2a": result_2a["u_slope"],
            "intercept_2a": result_2a["intercept"],
            "u_intercept_2a": result_2a["u_intercept"],

            "checks_2b": result_2b["checks"],
            "z_score_2b": result_2b["z_score"],
            "slope_2b": result_2b["slope"],
            "u_slope_2b": result_2b["u_slope"],
            "intercept_2b": result_2b["intercept"],
            "u_intercept_2b": result_2b["u_intercept"],

            "checks_2c": result_2c["checks"],
            "z_score_2c": result_2c["z_score"],
            "slope_2c": result_2c["slope"],
            "u_slope_2c": result_2c["u_slope"],
            "intercept_2c": result_2c["intercept"],
            "u_intercept_2c": result_2c["u_intercept"],
        })


        all_checks = []

        for f in ["1", "2", "2a", "2b", "2c"]:
            all_checks.extend(dictionary[f"checks_{f}"])

        dictionary["checks"] = np.unique(all_checks)  # storing all of the checks

        coincident_checks = []

        for check in all_checks:
            if all_checks.count(check) > 1:
                coincident_checks.append(check)

        dictionary["coincident_checks"] = np.unique(coincident_checks) # also storing the ones appering in various filters

# CREATION OF AN h5 FILE TO STORE THE RELATED DATA FOR THE INTERESTING SUBRUNS----------------------------------------------------------------------------------
        checks_h5 = tables.open_file(r"checks_Run{run}.h5".format(run=run), mode="w") #create a .h5 file for the subruns to check ('checks') for each run
        
        # In these files, we want to store:
        # checks_1 : subruns to check given by FILTER 1: EVENT RATE
        # checks_2 : subruns to check given by FILTER 2: EVENT RATE VS INTENSITY
        
        # and then, for each of those interesting subruns, store the following information (for reference check toy1.ipynb):
        # - All the information contained in the dl1datacheck file of the subrun
        # - For checks_f: the z_score
        
        checks_h5.create_group("/", "general", "General information of the checks") 
        checks_h5.create_group("/", "perfilter", "Information of the checks per filter")
        

        # create a group in the root of the file to store general information of the run
        
        # GENERAL GROUP---------------------------------------------------------------------------------------
        # we want to copy the 'cosmics', 'flatfield', 'pedestals' tables 
        
        checks = dictionary["checks"]
        tabs = ["cosmics", "flatfield", "pedestals"]
        
        for tab in tabs:

            if not hasattr(a.root.dl1datacheck, tab):
                print(f"Run {run}: missing table '{tab}', skipping.")
                continue
            
            origin_tab = a.root.dl1datacheck._f_get_child(tab)  # find that table within dl1datacheck
            origin_array = origin_tab[:]  # read the whole table as a numpy array
         
            filtered_data = origin_array[np.isin(origin_array['subrun_index'], checks)]
         
            t = checks_h5.create_table("/general", tab, origin_tab.description, "Table {tab} for checks".format(tab=tab))
            t.append(filtered_data)  # append all filtered rows at once, instead of row by row
            t.flush()

        class CoincidentChecks(tables.IsDescription):
            check_index = tables.Int32Col()

        t = checks_h5.create_table(
            "/general",
            "coincident_checks",
            CoincidentChecks,
            "Subruns selected by more than one filter"
        )

        row = t.row

        for check in dictionary["coincident_checks"]:
            row["check_index"] = check
            row.append()

        t.flush()        
        
        # PER FILTER GROUP---------------------------------------------------------------
        # for each filter (subgroup) store the checks and relevant parameters
        
        filters = ["1", "2", "2a", "2b", "2c"]

        sigma_cutoffs = {
            "1": sigma_cutoff_1,
            "2": sigma_cutoff_2,
            "2a": sigma_cutoff_2a,
            "2b": sigma_cutoff_2b,
            "2c": sigma_cutoff_2c,
        }
        
        class CheckResult(tables.IsDescription):
            check_index = tables.Int32Col()
            z_score = tables.Float64Col()
            sigma_cutoff = tables.Float64Col()
            fit_parameters = tables.Float64Col(shape=(4,))  # [slope, u_slope, intercept, u_intercept]
        
        
        for f in filters:
            t = checks_h5.create_table("/perfilter", "filter{f}".format(f=f), CheckResult, "Filter {f}".format(f=f))
            row = t.row
         
            checks_f = dictionary["checks_{f}".format(f=f)]
            z_score_f = dictionary["z_score_{f}".format(f=f)]
            fit_params = [
                dictionary["slope_{f}".format(f=f)],
                dictionary["u_slope_{f}".format(f=f)],
                dictionary["intercept_{f}".format(f=f)],
                dictionary["u_intercept_{f}".format(f=f)],
            ]
         
            for c, z in zip(checks_f, z_score_f):  #zip pairs up the checks with their corresponding zscores
                row["check_index"] = c
                row["z_score"] = z
                row["sigma_cutoff"] = sigma_cutoffs[f]
                row["fit_parameters"] = fit_params
                row.append()  #  we are writing in the same row all the data above. checks, zscore, sigmacutoff, fit params, 
         
            t.flush()
            
        a.close()
        checks_h5.close()

    print("Filter 1 could not be applied to Runs: ", disfunctional_1)
    print("Filter 2 could not be applied to Runs: ", disfunctional_2)

if __name__ == '__main__':
    main()







