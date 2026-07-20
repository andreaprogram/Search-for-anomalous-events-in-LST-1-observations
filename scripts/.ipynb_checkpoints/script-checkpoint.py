# FILTERS TO DETECT ANOMALIES

dictionary = {
    "checks": [],
    "slope_1": None,
    "u_slope_1": None,
    "intercept_1": None,
    "u_intercept_1": None,
    "checks_1": None,
    "z_score_1": None,
    "slope_2": None,
    "u_slope_2": None,
    "intercept_2": None,
    "u_intercept_2": None,
    "checks_2": None,
    "z_score_2": None,
}

import re
import tables
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from pprint import pprint

a = tables.open_file("/data/cta/users-ifae/moralejo/CTA/summer_student_2026/datacheck/datacheck_dl1_LST-1.Run24704.h5")

# USEFUL VARIABLES
subruns = (a.root.dl1datacheck.cosmics.col('subrun_index')) #0,1,...,57 number of subruns in this run
n_subruns = len(subruns)
time = a.root.dl1datacheck.cosmics.col('elapsed_time')


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

    for p in range(y.shape[1]):
        yp = y[:, p]
        model = sm.OLS(yp, Xc).fit()
        residual = np.abs(yp - model.predict(Xc))
        mad = np.median(np.abs(residual - np.median(residual)))
        mad = max(mad, 1e-12)  # avoid division by zero
        modified_z = 0.6745 * residual / mad
        good = modified_z <= z_thresh

        if good.sum() >= 2:
            model = sm.OLS(yp[good], Xc[good]).fit()
            fit[:, p] = model.predict(Xc)
            intercept[p] = model.params[0]
            slope[p] = model.params[1]
            u_intercept[p] = model.bse[0]
            u_slope[p] = model.bse[1]
        else:
            fit[:, p] = np.median(yp)
            intercept[p] = np.median(yp)
            slope[p] = 0.0
            u_intercept[p] = np.nan
            u_slope[p] = np.nan

        good_points.append((x[good], yp[good]))

    return {"fit": fit, "slope": slope, "u_slope": u_slope,
            "intercept": intercept, "u_intercept": u_intercept,
            "good_points": good_points}


# FILTER 1: EVENT RATE PER SUBRUN----------------------------------------------------------------------------------------------------
num_events = a.root.dl1datacheck.cosmics.col('num_events') # events in each subrun
event_rate = num_events / time
event_rate_sigma = np.sqrt(np.maximum(num_events, 1.0)) / time  

# Robust fit
event_rate_robfit = robust_fitt(subruns, event_rate.reshape(-1, 1))
event_rate_fit = event_rate_robfit["fit"].flatten()

# Selection criteria
erate_z_score = np.abs(event_rate - event_rate_fit) / event_rate_sigma
mask_1=erate_z_score > 3
checks_1 = subruns[mask_1]
z_score_1 = erate_z_score[mask_1]

dictionary.update({
    "checks_1": checks_1,
    "z_score_1": z_score_1,
    "slope_1": event_rate_robfit["slope"][0],
    "u_slope_1": event_rate_robfit["u_slope"][0],
    "intercept_1": event_rate_robfit["intercept"][0],
    "u_intercept_1": event_rate_robfit["u_intercept"][0]
})


# FILTER 2: EVENT RATE VS INTENSITY----------------------------------------------------------------------------------------------------
hist_intensity = a.root.dl1datacheck.cosmics.col('hist_intensity')

# Find the center of each bin for the histogram of intensity
intensity_bins = a.root.dl1datacheck.histogram_binning.col('hist_intensity')[0]
intensity = 0.5*(intensity_bins[1:]+intensity_bins[:-1])  

# Event rate per intensity value in each subrun
# Therefore, to perform a more sensitive analysis we'll implement the Kolmogorov-Smirnov statistic, because we want to compare the whole shape of our distribuition
# with that end we implement the Kolmogorov-Smirnov parameter D = max |F(x) - Fn| which tells us how different are two cumulative distribuition functions
# we will compare the CDF for each subrun with its neighbours, bc the satellite is expected to crossthe FoV within a fine period of time

cdf = np.cumsum(hist_intensity / hist_intensity.sum(axis=1, keepdims=True), axis=1) #we need to normalize the histogram. 1 cdf per subrun
D = np.zeros(n_subruns) 

for s in range(0, n_subruns-1):
    D[s] = np.max(np.abs(cdf[s] - cdf[s+1])) #we compare the CDF in each subrun 's' with its neighbouring one 's+1'

#DO A ROBUST FIT OF D TO SELECT THE GOOD POINTS
D_robust_fit = robust_fitt(subruns, D.reshape(-1, 1))

# This is not a Poisson process therefore we cannot treat it as such
D_good = D_robust_fit["good_points"][0][1]  # y-selected values for this column

# calculate the sigma for those good points so that it is not biased 
sigma_2 = np.std(D_good)
deviations_2 = np.abs(D - np.median(D_good))/sigma_2

# SUBRUNS TO CHECK
mask_2 = deviations_2 > 3
checks_2 = subruns[mask_2]
z_score_2 = deviations_2[mask_2]

dictionary.update({
    "checks_2": checks_2,
    "z_score_2": z_score_2,
    "slope_2": D_robust_fit["slope"][0],
    "u_slope_2": D_robust_fit["u_slope"][0],
    "intercept_2": D_robust_fit["intercept"][0],
    "u_intercept_2": D_robust_fit["u_intercept"][0],
})

dictionary["checks"] = np.unique(np.concatenate([
    np.asarray(dictionary["checks_1"], dtype=np.int64),
    np.asarray(dictionary["checks_2"], dtype=np.int64),
]))   #save all the checks together without repeating

#pprint(dictionary)





# CREATION OF AN h5 FILE TO STORE THE RELATED DATA FOR THE INTERESTING SUBRUNS


run = re.search(r"Run(\d+)", a.filename).group(0) #extract the run number from the filename, gropu(0) takes 'Run24704' and group(1) takes the thing in brackets '24704'
checks_h5 = tables.open_file(r"checks_{run}.h5".format(run=run), mode="w") #create a .h5 file for the subruns to check ('checks') for each run

# In these files, we want to store:
# checks_1 : subruns to check given by FILTER 1: EVENT RATE
# checks_2 : subruns to check given by FILTER 2: EVENT RATE VS INTENSITY

# and then, for each of those interesting subruns, store the following information (for reference check toy1.ipynb):
# - All the information contained in the dl1datacheck file of the subrun
# - For checks_f: the z_score

checks_h5.create_group("/", "global", "Global information of the checks") 
checks_h5.create_group("/", "perfilter", "Information of the checks per filter")


#create a group in the root of the file to store global information of the run

# GLOBAL GROUP---------------------------------------------------------------------------------------
# we want to copy the 'cosmics', 'flatfield', 'pedestals' tables 

checks = dictionary["checks"]
tabs = ["cosmics", "flatfield", "pedestals"]

for tab in tabs:
    origin_tab = a.root.dl1datacheck._f_get_child(tab)  # find that table within dl1datacheck
    origin_array = origin_tab[:]  # read the whole table as a numpy array
 
    filtered_data = origin_array[np.isin(origin_array['subrun_index'], checks)]
 
    t = checks_h5.create_table("/global", tab, origin_tab.description, "Table {tab} for checks".format(tab=tab))
    t.append(filtered_data)  # append all filtered rows at once, instead of row by row
    t.flush()
 
# PER FILTER GROUP---------------------------------------------------------------
# for each filter (subgroup) store the checks and relevant parameters
 
filters = [1, 2]  # in case we want to enlarge this list in the future
 
 
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
        row["sigma_cutoff"] = 3.0
        row["fit_parameters"] = fit_params
        row.append()  #  we are writing in the same row all the data above. checks, zscore, sigmacutoff, fit params, 
 
    t.flush()
 
a.close()
checks_h5.close()