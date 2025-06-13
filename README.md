# DTSeg_plotter

## Introduction
A quick method to plot the DTS EG and acquire relevant statistics in python

This is a simple script I threw together to plot the Diabetes Technology Societies error grid. The zones A - E were defined based on the quidelines laid out in the paper "The Diabetes Technology Society Error Grid and Trend Accuracy Matrix for Glucose Monitors" by David Klonoff et. al., 2024 (DOI: 10.1177/19322968241275701).

This is not intended to replace the official DTSeg plotter which can be found at https://www.diabetestechnology.org/dtseg/ but is just a useful tool to quickly plot your Monitor vs Reference data and acquire rough MARD, risk zone analysis and relative error information. 
There is a slight discrepancy between the official DTS plotter results and this version, as using the sample data provided on the website results in one fewer data point in zone A and one additional point in zone B. The cause for this is unknown, but if a better method of allocating points to zones, or just a better method is available then please improve this tool. 

##Instalation instructions 

To install this code simply type:
'''
pip install git+https://github.com/NIQS-Tech/DTSeg_plotter.git
'''

into your terminal in your chosen python environment.

##Usage instructions 

To use the DTS eg plotter you need to do type:
'''
from dtseg.dtseg import plot_DTSEG
'''
Calling the function without any data or other parameters 
'''
plot_DTSEG()
'''
will just simply return the blank error grid.


