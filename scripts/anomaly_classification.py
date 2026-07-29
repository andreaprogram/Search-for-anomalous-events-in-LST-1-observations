import glob 
import argparse  

import re
import tables

import numpy as np
import statsmodels.api as sm


# INSERTING THE PATH OF THE CHECKS ---------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="This is my code") # parser is the object that will read the arguments, description will appear when python prueba.py --help

parser.add_argument('-f', '--input_files', dest='input_files', required=True,
                    type=str, help='checks_RunXXXXX.h5 files to be processed')

def main():
    args = parser.parse_args() 
    file_list = sorted(glob.glob(args.input_files))

    # count the amount of total runs and the amount of interesting runs 
    total_runs = 0 
    interesting_runs= 0
    invalid_runs = 0
    normal_runs = 0
    
    filters = ["1", "2", "2a", "2b", "2c"]
    
    # lists to store the id of the runs for which our model is not valid
    disfunctional = {f: [] for f in filters} # for each filter, we store the invalid runs
    
    runs_per_filter = {f: [] for f in filters} #store the checks collected by each filter

    for file in file_list:
        total_runs += 1

        try:
            b = tables.open_file(file)       
            run_str = re.search(r"Run(\d+)", b.filename).group(1)  # saves correctly the run id's starting on 0, int() takes the numerical value
            run = int(run_str) #extract the run number from the filename, group(0) takes 'Run24704' and group(1) takes the thing in brackets '24704'

            checks = b.root.general.cosmics.col("subrun_index")
                
            validity = {}          
            
            for f in filters:
                fit_table = getattr(b.root.perfilter, f"filter{f}_fit")
                valid = fit_table[0]["valid"]
                validity[f] = valid # store the validity for every filter

                if not valid:
                    disfunctional[f].append(run)
                    
            # FIRST : see if this run has any interesting subruns--------------------------------------------
            if len(checks)==0:
                if not any(validity.values()): # if none of the filters are valid, it's an invalid run
                    invalid_runs += 1
                else:
                    normal_runs += 1  # if some filter is valid, it is just a normal run
                b.close()
                continue
                
            # SECOND : if the run has interesting subruns, anotate by which filter where those subruns detected-----------------------
            else:
                interesting_runs += 1
                
                for f in filters:
                    filter_table = getattr(b.root.perfilter, f"filter{f}")
                    
                    if len(filter_table)>0:
                        runs_per_filter[f].append(run)

            b.close()
            
        except Exception as e:
            print(f"Error procesando {file}: {e}")
            
            

    # CREATION OF AN h5 FILE TO STORE THE ANALYSIS OF THE RESULTS----------------------------------------------------------------------------------
    classifiedruns_h5 = tables.open_file(r"classified_runs.h5", mode="w") 
    
    # In these files, we want to store:
    #total_runs : total amount of runs within the 6 years
    #interesting_runs : interesting amount of runs within the total
    #invalid_runs : invalid amount of runs within the total
    #normal_runs : normal amount of runs within the total
    
    # and then, for each of those interesting runs, store the following information:
    # - by which filter they were marked as interesting

    # COUNTERS OF KINDS OF RUNS (saved as attributes)-------------------------------
    classifiedruns_h5.root._v_attrs.total_runs = total_runs
    classifiedruns_h5.root._v_attrs.interesting_runs = interesting_runs
    classifiedruns_h5.root._v_attrs.invalid_runs = invalid_runs
    classifiedruns_h5.root._v_attrs.normal_runs = normal_runs

    
    g_disfunctional = classifiedruns_h5.create_group("/", "disfunctional", "Invalid runs for all the filters") 
    g_interesting = classifiedruns_h5.create_group("/", "interesting", "Runs with interesting subruns to analyze")

    # LISTS WITH THE INTERESTING RUNS CLASSIFIED (saved as arrays)---------------------------------------
    for f in filters:
        classifiedruns_h5.create_array(g_disfunctional, f"filter_{f}", np.array(disfunctional[f], dtype=np.int32))
        classifiedruns_h5.create_array(g_interesting, f"filter_{f}", np.array(runs_per_filter[f], dtype=np.int32))

    classifiedruns_h5.close()

if __name__ == "__main__":
    main()