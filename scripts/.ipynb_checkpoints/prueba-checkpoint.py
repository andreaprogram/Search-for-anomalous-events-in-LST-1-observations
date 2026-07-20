import glob # to search files using paths
# import lstchain

import argparse  # to give an argument when executing the script
parser = argparse.ArgumentParser(description="This is my code") # parser is the object that will read the arguments, description will appear when python prueba.py --help

parser.add_argument('-f', '--input_files', dest='input_files', required=True,
                    type=str, help='Datacheck files to be processed')
# defining an argument (the 'input') 
# -f short way to call the file
# --input files long way
# required=True makes mandetory to provide an argument

def main():
    args = parser.parse_args()  #read arguments
    file_list = glob.glob(args.input_files)  # search for the input_files by the name

    for file in file_list:
        print(file)
        # Aquí todo el código que busca anomalías

if __name__ == '__main__':
    main()
