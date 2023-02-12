# Doppler-Spectroscopy

This program reads in and validates spectroscopy data from multiple data files by checking the files exist and filtering out non-numerical points or outliers. This program can operate if the orbit of the system isn't in the line of sight of the observer using an inclination factor.

The data is processed and a best fit sin curve is calculated by varying two parameters: the velocity of the star and the angular frequency of it's orbit. The initial values used to find these two parameters are calculated from the data points themselves and need not be entered by the user. The data, alongside this best fit line, is plotted and a plot of the residuals is also created for the user to see.

After completing this fit, mesh arrays are formed and used to create a contour plot which shows how the chi-squared parameter varies when the two fit parameters are varied. This contour plot also helps calculate the uncertainties on the two fit parameters.

Finally, the code propogates the two fit parameters along with their uncertainties to calculate and output several properties of the star-planet system. The program can also give the user a sense of how long a typical orbit would take and offers to write the calculated data into a text file.


Data files to be read in: 

'doppler_data_1.csv', 'doppler_data_2.csv'.
