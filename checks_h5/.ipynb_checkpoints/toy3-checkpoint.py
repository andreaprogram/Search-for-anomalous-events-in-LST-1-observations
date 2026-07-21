import glob # to search files using paths
import argparse  # to give an argument when executing the script

import re
import tables

from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe_io_lst import LSTEventSource

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
# CAMERA GEOMETRY------ 
sa = LSTEventSource.create_subarray(tel_id=1)
focal_length = sa.tel[1].optics.equivalent_focal_length
camera_geom = sa.tel[1].camera.geometry




def main():
    args = parser.parse_args()  #read arguments
    file_list = sorted(glob.glob(args.input_files))

    if args.batch_size is not None:
        start = args.batch * args.batch_size
        end = start + args.batch_size
        file_list = file_list[start:end]

    # lists to store the id of the runs for which our model is not valid
    disfunctional_3 = []

    #invalid_runs = [np.int32(20908), np.int32(21187), np.int32(21188), np.int32(21387), np.int32(21467), np.int32(21474), np.int32(21509), np.int32(21778), np.int32(21818), np.int32(21877), np.int32(22259), np.int32(22685), np.int32(22737), np.int32(23045), np.int32(20735)]

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

        # FILTER 3: FLUCTUATIONS OF CoG WITHIN PIXEL--------------------------------------------------------------------------------------------------
        # we expect the cog within pixel rate to increase with respect to its tendency in the subrun where we have the anomaly in the pixels by which it crosses
        cog_pixel = a.root.dl1datacheck.cosmics.col('cog_within_pixel')[:-1]  # shape: (n_subruns, n_pixels)
        
        cog_rate = cog_pixel / time[:, None]
        cog_sigma = np.sqrt(np.maximum(cog_pixel, 1.0)) / time[:, None] # Poisson-like uncertainty associated to each value
        
        
        # 3.1. ROBUST FIT TO SELECT PIXELS IN WHICH THE ANOMALY HAS FALLEN 
        cog_robfit = robust_fitt(subruns, cog_rate)  # we are contructing a linear robust fit for each pixel along all the subruns in the run

        if not cog_robfit["valid"]:
            disfunctional_3.append(run)
            
            dictionary.update({
                    "checks": np.array([], dtype=np.int64),
                    "z_score": np.array([], dtype=np.float64),
                    "slope": np.nan,
                    "u_slope": np.nan,
                    "intercept": np.nan,
                    "u_intercept": np.nan})

        else:
            cog_fit = robust_fitt(subruns, cog_rate)["fit"]
        
            # Selection criteria
            sigma_cutoff_3 = 3
            cog_z_score = (cog_rate - cog_fit) / cog_sigma
            mask_3 = (cog_z_score) > sigma_cutoff_3 # when taking np.abs(z_score), much more anomalous pixels appear
            
            # SELECTION OF THE PIXELS
            # Select subrun interesting to check according to this filter
            checks_3 = subruns[np.argmax(mask_3.sum(axis=1))]
            
            dictionary.update({
                    "checks": checks_3,
                    "z_score": cog_z_score,
                    "slope": cog_robfit["slope"][0],
                    "u_slope": cog_robfit["u_slope"][0],
                    "intercept": cog_robfit["intercept"][0],
                    "u_intercept": cog_robfit["u_intercept"][0]})            

            # 3.2 : DETECTION OF POTENTIAL SATELLITES/METEORITES BY IDENTIFYING LINEAR TRACES
            # we expect bodies such as satellites (our main interest) or meteorites to cross the camera FoV in a straight trajectory
            # to check this, having identified the 'affected' pixels, we aim to do a linear fit (PCA) and determine whether the selected pixels do form a straight line via a Chi^2 test 
            
            # Linear fit to check that indeed the path is linear, we use the engineering camera frame to represent it later: x=-y y=-x
            x_pixels = camera_geom.transform_to(EngineeringCameraFrame()).pix_x[mask_3[checks_3]].value  # .value to remove the units in the array
            y_pixels = camera_geom.transform_to(EngineeringCameraFrame()).pix_y[mask_3[checks_3]].value
            
            pixels_fit = robust_fitt(x_pixels, y_pixels.reshape(-1, 1))
            x_fit = np.linspace(min(x_pixels), max(x_pixels), 100)
            









        
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
