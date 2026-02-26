#!/bin/bash

# $1 and $2 are special variables in bash that contain the 1st and 2nd 
# command line arguments to the script, which are the names of the
# Dakota parameters and results files, respectively.

params=$1
results=$2

############################################################################### 
##
## Pre-processing Phase -- Generate/configure an input file for your simulation 
##  by substiting in parameter values from the Dakota paramters file.
##
###############################################################################

# dprepro $params cantilever.template cantilever.i
#/home/minamoto/Desktop/gphase_dakota/dprepro $params Gver6.14.TDB.temp Gver6.14.TDB 
./dprepro $params Gver.6.53.TDB.temp Gver.6.53.TDB 

############################################################################### 
##
## Execution Phase -- Run your simulation
##
###############################################################################


#python3 < /home/minamoto/Desktop/gphase_dakota/tc_exec.py   
python3 < ./tc_exec.py   

############################################################################### 
##
## Post-processing Phase -- Extract (or calculate) quantities of interest
##  from your simulation's output and write them to a properly-formatted
##  Dakota results file.
##
###############################################################################

#mass=$(tail -15 cantilever.log | head -1 | awk '{print $1}')
#stress=$(tail -11 cantilever.log | head -1 | awk '{print $1}')
error_tot=$(tail -7 error_tot.txt | head -1 | awk '{print $1}')

#echo "$mass mass" > $results
#echo "$stress stress" >> $results
#echo "$error_tot error_tot" >> $results
echo "$error_tot error_total" > $results

