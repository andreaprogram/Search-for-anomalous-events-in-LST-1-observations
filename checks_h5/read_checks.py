# THE AIM OF THIS CODE IS TO EASILY READ THE INFORMATION PROVIDED BY checks_h5 FILES
import glob 
import argparse  
import tables 
import re

parser = argparse.ArgumentParser(description="This is my code") 
parser.add_argument('-f', '--input_files', dest='input_files', required=True,
                    type=str, help='Datacheck files to be processed')
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

def main():
    args = parser.parse_args()  
    file_list = sorted(glob.glob(args.input_files))

    if args.batch_size is not None:
        start = args.batch * args.batch_size
        end = start + args.batch_size
        file_list = file_list[start:end]
    
    for file in file_list:
        print(re.search(r"Run(\d+)", file).group(0))
        
        b = tables.open_file(file)
        
        for filt in ["filter1", "filter2"]:
    
            table = getattr(b.root.perfilter, filt)
            
            print(f"\n{'='*60}")
            print(f"{filt.upper()}")
            print(f"{'='*60}")
            
            if len(table) == 0:
                print("No checks found.")
                continue
            
            # Los parámetros del fit son iguales para todas las filas
            fit = table[0]["fit_parameters"]
            slope, u_slope, intercept, u_intercept = fit
            
            print("Fit parameters:")
            print(f"  slope     = {slope:.6f} ± {u_slope:.6f}")
            print(f"  intercept = {intercept:.6f} ± {u_intercept:.6f}")
            
            print("\nChecks:")
            print(f"{'Subrun':>8} {'z-score':>10}")
            print("-" * 20)
            
            for row in table:
                print(f"{row['check_index']:>8} {row['z_score']:>10.3f}")

        b.close()

if __name__ == '__main__':
    main()

