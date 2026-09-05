This project focuses on the search of 'anomalous' events in the CTAO's LST-1 telescope observations, such as: satellites (our main interest), meteorites, car flashes or other undetermined objects. With that end, the development of a detection method for such events has been carried out, under the supervision of Dr. Abelardo Moralejo at IFAE. 

In this repository can be found all of the code developed for the project, most importantly: 

- toy1.ipynb*: A very first draft on what possible filters to apply to the observational data stored in datacheck files, with the goal to obtain as an output in which instants of time (subruns) an anomalous event might have ocurred.
- toy2.py: The "final version" of toy1. Filters 1 & 2 are executed and an .h5 file with the anomalous subruns and their interesting information. 
- zscores_calc.py and zscores_notebook.ipynb:  to determine statistically sigma_cutoff 1 & 2 used in toy2
- toy3.py: Taking the output of toy2 as an input, toy3 executes a third filter with the aim to find straight paths among the anomalies.
- chis2.py and chis2_notebook.ipynb: to determine statistically sigma_pixel used in toy2 
- satellite_identification.ipynb: to identificate the object observed in the anomalous subrun, given the observation conditions.
- anomaly_classification.py: to classify the outputs of toy2 and later perform an statistical analyisis of the results.
- filters_results.ipynb: to analyse the obtained results from toy2 and toy3.

*Jokingly named 'toys' as they are simple tools that allow us to start the study of these anomalies in the observations. 
