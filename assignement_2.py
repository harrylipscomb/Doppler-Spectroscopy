# -*- coding: utf-8 -*-
"""
________________TITLE___________________
PHYS20161 - Assignment 2 - Doppler Spectroscopy
----------------------------------------
This program reads in and validates spectroscopy data from multiple data files
by checking the files exist and filtering out non-numerical points or
outliers. This program can operate if the orbit of the system isn't in the line
of sight of the observer using an inclination factor.

The data is processed and a best fit sin curve is calculated by varying two
parameters: the velocity of the star and the angular frequency of it's orbit.
The initial values used to find these two parameters are calculated from the
data points themselves and need not be entered by the user. The data, alongside
this best fit line, is plotted and a plot of the residuals is also created for
the user to see.

After completing this fit, mesh arrays are formed and used to create a contour
plot which shows how the chi-squared parameter varies when the two fit
parameters are varied. This contour plot also helps calculate the uncertainties
on the two fit parameters.

Finally, the code propogates the two fit parameters along with their
uncertainties to calculate and output several properties of the star-planet
system. The program can also give the user a sense of how long a typical
orbit would take and offers to write the calculated data into a text file.

Last Updated: 16/12/20
@author: Harry Lipscomb, Student ID: 10449626
"""
from time import sleep as pause
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.constants as pc
from tqdm import tqdm as counter

SECONDS_TO_YEARS = 1 / 31536000
WAVELENGTH_EMITTED = 656.281 * 10**-9 #Wavelength emitted in m.
STAR_MASS = 2.78 * 2 * 10**30
SPEED_OF_LIGHT = pc.c
GRAVITATION_CONSTANT = pc.G
METRES_TO_AU = 1 / pc.astronomical_unit
KG_TO_JOVIAN = 1 / (1.89813 * 10**27)
FILE_1 = 'doppler_data_1.csv'
FILE_2 = 'doppler_data_2.csv'

INCLINATION = ((np.pi) / 2)
"""Inclination should be added in radians. The value should be between -pi/2
and +pi/2. For more extreme inclinations the phase of the sin wave being fit
should be changed accordingly."""

def check_data_files(file_1, file_2):
    """
    Checks if both the data files to be read exist.

    Parameters
    ----------
    file_1 : Comma-Separated Values File (CSV)
        First data file.
    file_2 : Comma-Separated Values File (CSV)
        Second data file.

    Returns
    -------
    bool
        Returns True if files exist and False if a FileNotFoundError occurs.
    """
    try:
        file_1 = open('doppler_data_1.csv', 'r')
        file_1.close()
    except FileNotFoundError:
        print("The first data file wasn't found. Check file named correctly.")
        return False
    try:
        file_2 = open('doppler_data_2.csv', 'r')
        file_2.close()
    except FileNotFoundError:
        print("The second data file wasn't found. Check file named correctly.")
        return False
    return True

def read_in_data(file_1, file_2):
    """
    Reads in and orders the two data files. Uses the 'is_valid' function to
    validate the data and deletes any unwanted data. Splits up each column of
    the remaining data into individual numpy arrays then checks for and deals
    with any large outliers in the data using the 'is_large_outlier' function.

    Parameters
    ----------
    file_1 : Comma-Separated Values File Name (CSV)
        First data file.
    file_2 : Comma-Separated Values File Name (CSV)
        Second data file.

    Returns
    -------
    times : numpy array
        Times from both files in numerical order.
    wavelengths : numpy array
        Wavelengths of star's light from both files in time order.
    wavelength_uncertainties : numpy array
        Uncertainty in wavelengths of light from both files in time order.

    """
    file_1_data = np.genfromtxt(file_1, delimiter=',', comments='%')
    file_2_data = np.genfromtxt(file_2, delimiter=',', comments='%')
    data_set = np.vstack((file_1_data, file_2_data))
    data_set_sorted = data_set[data_set[:, 0].argsort()]
    indexs_to_delete = []
    for index, data in enumerate(data_set_sorted[:, 0]):
        if is_valid(data) is False:
            indexs_to_delete.append(index)
    for index, data in enumerate(data_set_sorted[:, 1]):
        if is_valid(data) is False:
            indexs_to_delete.append(index)
    for index, data in enumerate(data_set_sorted[:, 2]):
        if is_valid(data) is False:
            indexs_to_delete.append(index)

    data_set_sorted = np.delete(data_set_sorted, indexs_to_delete, axis=0)
    times = data_set_sorted[:, 0]
    wavelengths = data_set_sorted[:, 1]
    wavelength_uncertainties = data_set_sorted[:, 2]
    [times, wavelengths,
     wavelength_uncertainties] = is_large_outlier(times,
                                                  wavelengths,
                                                  wavelength_uncertainties)

    return (times, wavelengths, wavelength_uncertainties)

def is_valid(entry):
    """
    Checks if each time, wavelength and wavelength uncertainty is numerical.

    Parameters
    ----------
    entry : numerical or non-numerical
        This entry can be either a time, wavelength or wavelength uncertainty.

    Returns
    -------
    bool
        True if the entry is a numerical value and false if it's not a number.

    """
    try:
        entry = float(entry)
        if np.isnan(entry):
            return False
    except ValueError:
        return False
    return True

def wavelength_to_velocity(wavelengths, wavelength_emitted, speed_of_light,
                           inclination):
    """
    Converts each wavelength value to a velocity for the star relative to
    Earth.

    Parameters
    ----------
    wavelengths : numpy array
        Wavelengths of light from star.
    wavelength_emitted : float
        Known wavelength of light emitted from star.
    speed_of_light : float
        The universal speed of light.
    inclination: float
        The angle between the normal of the orbital plane of the system and the
        observer's line of sight in radians.

    Returns
    -------
    velocities : numpy array
        An array of star velocities.

    """
    velocities = ((((wavelengths / wavelength_emitted) - 1) * speed_of_light)
                  / np.sin(inclination))

    return velocities

def wavelength_unc_to_velocity_unc(wavelength_uncertainties, speed_of_light,
                                   wavelength_emitted, inclination):
    """
    Converts each wavelength uncertainty to a velocity uncertainty.

    Parameters
    ----------
    wavelength_uncertainties : numpy array
        Array of wavelength uncertainties.
    speed_of_light : float
        The universal speed of light.
    wavelength_emitted : float
        Known wavelength of light emitted from star.
    inclination: float
        The angle between the normal of the orbital plane of the system and the
        observer's line of sight in radians.

    Returns
    -------
    velocity_uncertainties : numpy array
        Array of star velocity uncertainties.

    """
    velocity_uncertainties = (wavelength_uncertainties * speed_of_light) / (
        wavelength_emitted * np.sin(inclination))

    return velocity_uncertainties

def is_large_outlier(times, wavelengths, wavelength_uncertainties):
    """
    Check for large outliers in the data by seeing if each wavelength is more
    than 5 standard deviations away from the mean wavelength.

    Parameters
    ----------
    times : numpy array
        Times for each data point in years.
    wavelengths : numpy array
        Wavelengths of light from star.
    wavelength_uncertainties : numpy array
        Wavelength uncertainties on light from star.

    Returns
    -------
    times : numpy array
        Array of times excluding large outliers.
    wavelengths : numpy array
        Array of wavelengths excluding large outliers.
    wavelength_uncertainties : numpy array
        Array of wavelength uncertainties excluding large outliers.

    """
    for index, wavelength in enumerate(wavelengths):
        if np.absolute(wavelength - np.mean(wavelengths)) > (
                5 * np.std(wavelengths)):
            times = np.delete(times, index)
            wavelengths = np.delete(wavelengths, index)
            wavelength_uncertainties = np.delete(wavelength_uncertainties,
                                                 index)

    return (times, wavelengths, wavelength_uncertainties)

def chi_squared(xy_values):
    """
    Defines our chi-squared function to be minimised. Creates a prediction
    array and compares this to the velocity data points to output a chi
    squared value.

    Parameters
    ----------
    xy_values : tuple
        Contains a tuple of both the velocity constant and angular frequency.

    Returns
    -------
    chi_squared_value : float
        The calculated chi square result for that set of parameters.

    """
    [velocity_constant, angular_frequency] = xy_values
    prediction = function_to_fit(velocity_constant, angular_frequency, TIMES)
    chi_squared_value = 0
    for element in range(0, len(TIMES)):
        if VELOCITY_UNCERTAINTIES[element] != 0:
            chi_squared_value += ((prediction[element] - VELOCITIES[element]) /
                                  VELOCITY_UNCERTAINTIES[element]) ** 2

    return chi_squared_value

def is_small_outlier(velocity_minimum, angular_frequency_minimum, times,
                     velocities, velocity_uncertainties):
    """
    Check if each wavelength is an outlier with relation to the line of best
    fit. Discards any data points which have either no uncertainty or are more
    than 3 standard deviations away from the best fit line.

    Parameters
    ----------
    velocity_minimum : float
        The velocity which achieves the minimum chi-squared fit line before
        small outliers are discarded.
    angular_frequency_minimum : float
        The angular frequency which achieves the minimum chi-squared fit line
        before small outliers are discarded.
    times : numpy array
        Array of time values with small outliers.
    velocities : numpy array
        Array of velocities with small outliers.
    velocity_uncertainties : numpy array
        Array of velocity uncertainties with small outliers.

    Returns
    -------
    times : numpy array
        An array of times for each data point without small outliers.
    velocities : numpy array
        An array of star velocity data points without small outliers.
    velocity_uncertainties : numpy array
        An array of velocity uncertainties without small outliers.

    """
    predicted_velocities = function_to_fit(velocity_minimum,
                                           angular_frequency_minimum, times)
    indexs_to_delete = []
    for index in range(0, len(times)):
        if velocity_uncertainties[index] == 0:
            indexs_to_delete.append(index)
        else:
            if np.absolute(predicted_velocities[index] - velocities[index]) > (
                    3 * velocity_uncertainties[index]):
                indexs_to_delete.append(index)

    velocities = np.delete(velocities, indexs_to_delete)
    velocity_uncertainties = np.delete(velocity_uncertainties,
                                       indexs_to_delete)
    times = np.delete(times, indexs_to_delete)

    return (times, velocities, velocity_uncertainties)

def initial_parameter_calculator(velocities, times):
    """
    Calculates an initial estimate of the star's velocity and angular frequency
    to input into the minimization function. This assumes the magnitude of the
    angular frequency of the planet-star system is within the magnitudes of
    Pluto's and Mercury's angular frequencies.

    Parameters
    ----------
    velocities : numpy array
        Array of velocity data points for the star.
    times: numpy array
        Array of times for each data point.

    Returns
    -------
    velocity_estimate : float
        First estimate at the star's velocity.
    angular_frequency_estimate: float
        First estimate of the systems angular frequency.

    """
    initial_velocity_total = np.sum(((velocities ** 2)**(1/2)) * np.sqrt(2))
    velocity_estimate = initial_velocity_total / len(velocities)

    minimum_angular_frequency = 6.31 * 10**(-3) #An 1000 year orbit in rad/yr.
    maximum_angular_frequency = 230.21 #A 10 day orbit in rad/yr.
    angular_frequencies = np.geomspace(maximum_angular_frequency,
                                       minimum_angular_frequency,
                                       num=len(times))
    chi_squareds = np.array([])
    for frequency in angular_frequencies:
        chi_squareds = np.append(chi_squareds, chi_squared((velocity_estimate,
                                                            frequency)))
    minimum_chi = np.min(chi_squareds)
    index_of_minimum = np.where(chi_squareds == minimum_chi)
    angular_frequency_estimate = angular_frequencies[index_of_minimum]

    return (velocity_estimate, angular_frequency_estimate)

def function_to_fit(velocity_constant, angular_frequency, time_values):
    """
    Defines the sin wave function which will be minimised to find the desired
    parameters.

    Parameters
    ----------
    velocity_constant : float
        The velocity of the star to be considered.
    angular_frequency : float
        The angular frequency of the star-planet system to be considered.
    time_values : numpy array
        Array of time values for the star.

    Returns
    -------
    velocity_values : numpy array
        An array of the star's velocity values at each time value.

    """
    velocity_values = velocity_constant * np.sin((
        angular_frequency * time_values) + np.pi)

    return velocity_values

def fitline_against_data(velocity_minimum, angular_frequency_minimum, times,
                         velocities, velocity_uncertainties):
    """
    Plots a line of best fit using the minimised chi-squared parameters found
    next to the individual velocities. Also calculates the residuals for the
    data against the best fit line and plots this below on the same figure.

    Parameters
    ----------
    velocity_minimum : float
        Velocity value which results in the minimum chi-squared value.
    angular_frequency_minimum : float
        Angular frequency value which results in the minimum chi-squared value.
    times : numpy array
        Array of time values for the star.
    velocities : numpy array
        Array of star velocities.
    velocity_uncertainties : numpy array
        Array of star velocity uncertainties.

    Returns
    -------
    None.

    """
    time_values = np.linspace(0, np.max(times), 100)
    even_spaced_velocities = function_to_fit(velocity_minimum,
                                             angular_frequency_minimum,
                                             time_values)
    figure = plt.figure(figsize=(8, 6))
    fit_of_velocities = function_to_fit(velocity_minimum,
                                        angular_frequency_minimum, times)
    residuals = velocities - fit_of_velocities
    empty_array = np.zeros((len(times)))

    axes_1 = figure.add_subplot(211)
    axes_1.plot(time_values, even_spaced_velocities, dashes=[3, 2], color='r',
                label='Best Fit Line')
    axes_1.errorbar(times, velocities, yerr=velocity_uncertainties, fmt='.',
                    color='g', label='Data Points')
    axes_2 = figure.add_subplot(313)
    axes_2.plot(times, empty_array, dashes=[1, 2], color='r')
    axes_2.errorbar(times, residuals, yerr=velocity_uncertainties, fmt='.',
                    color='g')

    axes_1.set_title('Recorded Velocities against Time')
    axes_2.set_title('Plot of Residuals')
    axes_1.set_ylabel('Velocity $[m/s]$')
    axes_2.set_ylabel('Velocity $[m/s]$')
    axes_1.set_xlabel('Time $[yrs]$')
    axes_2.set_xlabel('Time $[yrs]$')
    axes_1.legend()
    plt.savefig('fitline_against_data.png', dpi=300)
    plt.show()

def orbital_distance(angular_frequency_minimum):
    """
    Calculates the orbital distance of the system from the angular frequency.

    Parameters
    ----------
    angular_frequency_minimum : float
        The frequency which results in the minimum chi-squared value.

    Returns
    -------
    orbit_distance_AU : float
        The distance of the planet to the star in astronomical units.

    """
    orbit_distance_metres = ((GRAVITATION_CONSTANT * STAR_MASS) /
                             angular_frequency_minimum**2)**(1/3)
    orbit_distance_au = orbit_distance_metres * METRES_TO_AU

    return orbit_distance_au

def planet_velocity(orbit_distance):
    """
    Determines the velocity of the planet from it's orbit distance.

    Parameters
    ----------
    orbit_distance : float
        Orbit distance of the planet.

    Returns
    -------
    planet_velocity_value : float
        The velocity of the planet in metres per second.

    """
    planet_velocity_value = ((GRAVITATION_CONSTANT * STAR_MASS)
                             / orbit_distance)**(1/2)

    return planet_velocity_value

def mass_planet(velocity_minimum, velocity_planet):
    """
    Determines the mass of the planet from it's velocity and the velocity of
    the star. Converts this to Jovian masses.

    Parameters
    ----------
    velocity_minimum : float
        Velocity of the star which minimises the chi-squared parameter.
    velocity_planet : float
        Calculated velocity of the planet.

    Returns
    -------
    mass_planet_jovian : float
        The mass of the planet in Jovian masses.

    """
    mass_planet_kg = (STAR_MASS * velocity_minimum) / velocity_planet
    mass_planet_jovian = mass_planet_kg * KG_TO_JOVIAN

    return mass_planet_jovian

def orbit_distance_uncertainty(angular_frequency_uncertainty,
                               angular_frequency_minimum):
    """
    Calculates the orbital distance's uncertainty. Converts this to
    astronomical units.

    Parameters
    ----------
    angular_frequency_uncertainty : float
        Uncertainty on the angular frequency which minimises chi-squared.
    angular_frequency_minimum : float
        Angular frequency which minimises the chi-squared parameter.

    Returns
    -------
    distance_uncertainty_au : float
        Uncertainty in orbital distance in astronomical units.

    """
    distance_uncertainty_metres = angular_frequency_uncertainty * (2/3) * (((
        GRAVITATION_CONSTANT * STAR_MASS) / angular_frequency_minimum**5)**(
            1/3))
    distance_uncertainty_au = distance_uncertainty_metres * METRES_TO_AU

    return distance_uncertainty_au

def planet_velocity_uncertainty(orbital_distance_uncertainty, orbit_distance):
    """
    Calculates the uncertainty in the planet's velocity.

    Parameters
    ----------
    orbital_distance_uncertainty : float
        Uncertainty on the orbital distance of the planet.
    orbit_distance : float
        Orbital distance of the planet.

    Returns
    -------
    velocity_uncertainty : float
        Uncertainty in the planet's velocity in metres per second.

    """
    velocity_uncertainty = (1 / (2 * (orbit_distance**(3/2)))) * ((
        GRAVITATION_CONSTANT * STAR_MASS)**(1/2)) * orbital_distance_uncertainty

    return velocity_uncertainty

def mass_planet_uncertainty(velocity_star, velocity_planet,
                            velocity_star_uncertainty,
                            velocity_planet_uncertainty, planet_mass):
    """
    Determines the uncertainty on the mass of the planet.

    Parameters
    ----------
    velocity_star : float
        The velocity of the star which minimises the chi-squared parameter.
    velocity_planet : float
        The caluclated planet velocity.
    velocity_star_uncertainty : float
        The uncertainty on the velocity of the star.
    velocity_planet_uncertainty : float
        The uncertainty on the velocity of the planet.
    planet_mass : float
        Mass of the planet in kilograms.

    Returns
    -------
    planet_mass_uncertainty_jovian : float
        Uncertainty on the mass of the planet in Jovian masses.

    """
    planet_mass_uncertainty = ((((velocity_star_uncertainty / velocity_star)**2)
                                + (velocity_planet_uncertainty / velocity_planet)**2)
                               **0.5) * planet_mass
    planet_mass_uncertainty_jovians = planet_mass_uncertainty * KG_TO_JOVIAN

    return planet_mass_uncertainty_jovians

def mesh_arrays(velocity_minimum, angular_freq_minimum, times):
    """
    Forms two square arrays in order to create a contour plot of the
    chi-squared parameter.

    Parameters
    ----------
    velocity_minimum : float
        Velocity which minimises the chi-squared parameter.
    angular_freq_minimum : float
        The angular frequency which minimises the chi-squared parameter.
    times: numpy array
        Array of times for each data point.

    Returns
    -------
    angular_frequency_mesh : numpy array
        Array of angular frequencies to use as the x input in a contour plot.
    velocity_mesh : numpy array
        Array of velocities to use as the y input in a contour plot.

    """
    velocity_values = np.linspace(
        (1 / 1.1) * velocity_minimum, 1.1 * velocity_minimum, len(times))
    angular_freq_minimum = angular_freq_minimum * (1 / SECONDS_TO_YEARS)
    upper_x_boundary = angular_freq_minimum * 1.1
    lower_x_boundary = angular_freq_minimum * (1 / 1.1)
    angular_frequency_values = np.linspace(
        lower_x_boundary, upper_x_boundary, len(times))
    angular_frequency_mesh = np.empty((0, len(angular_frequency_values)))
    for dummy_value in velocity_values:
        angular_frequency_mesh = np.vstack((angular_frequency_mesh,
                                            angular_frequency_values))
    velocity_mesh = np.empty((0, len(velocity_values)))
    for dummy_value in angular_frequency_values:
        velocity_mesh = np.vstack((velocity_mesh, velocity_values))
    velocity_mesh = np.transpose(velocity_mesh)

    return (angular_frequency_mesh, velocity_mesh)

def chi_squared_array(angular_frequency_mesh, velocity_mesh, times):
    """
    Forms the z component of the contour plot. This is the chi-squared value
    at each pair of parameters.

    Parameters
    ----------
    angular_frequency_mesh : numpy array
        A square mesh of angular frequencies centred around the minimum.
    velocity_mesh : numpy array
        A square mesh of velocities centred around the minimum.
    times: numpy array
        Array of times for each data point.

    Returns
    -------
    chi_array : numpy array
        Square array with the same dimensions as the two inputted arrays. Holds
        the chi-squared value for various velocity, angular frequency pairs.

    """
    chi_array = np.empty([len(times), len(times)])
    for i in range(0, len(times)):
        for j in range(0, len(times)):
            chi_array[i][j] = chi_squared((
                velocity_mesh[i][j], angular_frequency_mesh[i][j]))

    return chi_array

def reading_and_preparing_data(file_1, file_2, inclination):
    """
    Gathers several of the previously defined functions to read in and prepare
    the data and its uncertainty.

    Parameters
    ----------
    file_1 : Comma-Separated Values File (CSV)
        First file of data to be used.
    file_2 : Comma-Separated Values File (CSV)
        Second file of data to be used.
    inclination: float
        The angle between the normal of the orbital plane of the system and the
        observer's line of sight in radians.

    Returns
    -------
    times : numpy array
        Array of times for each data point.
    velocities : numpy array
        Array of star velocities
    velocity_uncertainties : numpy array
        Array of star velocity uncertainties.

    """
    times, wavelengths, wavelength_uncertainties = read_in_data(file_1, file_2)
    wavelengths = wavelengths / 10**9
    wavelength_uncertainties = wavelength_uncertainties / 10**9 #Convert nm to m
    velocities = wavelength_to_velocity(
        wavelengths, WAVELENGTH_EMITTED, SPEED_OF_LIGHT, inclination)
    velocity_uncertainties = wavelength_unc_to_velocity_unc(
        wavelength_uncertainties, SPEED_OF_LIGHT, WAVELENGTH_EMITTED, inclination)

    return (times, velocities, velocity_uncertainties)

def fit_with_outliers(times, velocities, velocity_uncertainties):
    """
    Makes an initial fit of the data points. Uses this first minimisation
    of chi-squared to filter any smaller outliers in the data.

    Parameters
    ----------
    times : numpy array
        Array of times with large outliers deleted.
    velocities : numpy array
        Array of velocities with large outliers deleted.
    velocity_uncertainties : numpy array
        Array of velocity uncertainties with large outliers deleted.

    Returns
    -------
    times : numpy array
        Array of times with small outliers deleted.
    velocities : numpy array
        Array of velocities with small outliers deleted.
    velocity_uncertainties : numpy array
        Array of velocity uncertainties with small outliers deleted.
    velocity_minimum: float
        Velocity for which the chi-squared value is minimisied for the data
        with small outliers.
    angular_frequency_minimum: float
        The angular frequency for which the chi-squared value is minimised for
        the data with small outliers.
    """
    initial_velocity, initial_frequency = initial_parameter_calculator(
        velocities, times)
    initial_frequency = float(initial_frequency)
    [velocity_minimum, angular_frequency_minimum] = fmin(
        chi_squared, (initial_velocity, initial_frequency),
        full_output=True)[0]
    times, velocities, velocity_uncertainties = is_small_outlier(
        velocity_minimum, angular_frequency_minimum, times, velocities,
        velocity_uncertainties)

    return (times, velocities, velocity_uncertainties, velocity_minimum,
            angular_frequency_minimum)

def second_fit_plot(initial_velocity, initial_angular_frequency, times,
                    velocities, velocity_uncertainties):
    """
    Completes a second chi-squared minimisation after the small outliers have
    been deleted. Also graphs this best fit line against the data and
    produces a residual plot.

    Parameters
    ----------
    initial_velocity : float
        Uses the velocity found in the first fit as the estimate of the star's
        velocity.
    initial_angular_frequency : float
        Uses the angular frequency found during the first fit as an estimate
        for the systems angular frequency.
    times : numpy array
        Array of times with small outliers deleted.
    velocities : numpy array
        Array of velocities with small outliers deleted.
    velocity_uncertainties : numpy array
        Array of velocity uncertainties with small outliers deleted.

    Returns
    -------
    velocity_minimum : float
        Star's velocity which minimises the chi-squared parameter.
    angular_frequency_minimum : float
        Angular frequency of system which minimises the chi-squared parameter.
    reduced_chi_squared : float
        The reduced chi-squared value for the fit with no outliers.

    """
    fit_results = fmin(chi_squared, (initial_velocity,
                                     initial_angular_frequency),
                       full_output=True)
    [velocity_minimum, angular_frequency_minimum] = fit_results[0]
    fitline_against_data(velocity_minimum, angular_frequency_minimum, times,
                         velocities, velocity_uncertainties)
    chi_squared_minimum = fit_results[1]
    degrees_of_freedom = len(times) - 2 #Two parameters are fit so minus 2.
    reduced_chi_squared = chi_squared_minimum / degrees_of_freedom

    return (velocity_minimum, angular_frequency_minimum, reduced_chi_squared)

def contour_plot_and_uncertainties(angular_frequency_years, velocity_minimum,
                                   chi_squared_value, times):
    """
    Finds the uncertainties on the two fit parameters. Forms the mesh arrays
    and uses them to create the contour plot. The uncertainty points are also
    shown on the plot.

    Parameters
    ----------
    angular_frequency_minimum : float
        Angular frequency which minimises the chi-squared parameter.
    velocity_minimum : float
        The velocity of the star which minimises the chi-squared parameter.
    chi_squared_value : float
        Minimum chi-squared value possible from data.

    Returns
    -------
    velocity_uncertainty : float
        Uncertainty on the velocity of the star in metres per second.
    angular_frequency_uncertainty : float
        Uncertainty on the angular frequency of the system in radians per year.

    """
    [angular_frequency_mesh,
     velocity_mesh] = mesh_arrays(velocity_minimum,
                                  angular_frequency_years * SECONDS_TO_YEARS,
                                  times)
    chi_array = chi_squared_array(angular_frequency_mesh, velocity_mesh, times)

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111)
    contour_plot = axis.contour(angular_frequency_mesh, velocity_mesh,
                                chi_array, 10)
    axis.clabel(contour_plot, inline=1, fontsize=10, fmt='%2.1f')
    get_points = axis.contour(angular_frequency_mesh, velocity_mesh,
                              chi_array, levels=[chi_squared_value + 1],
                              linestyles='dashed')
    get_points.levels[0] = chi_squared_value + 1
    axis.clabel(get_points, get_points.levels[:], inline=1, fontsize=9,
                fmt='%1.1f') #Above the contours are plotted and labelled.
    angular_frequencies = get_points.allsegs[0][0][:, 0]
    velocities = get_points.allsegs[0][0][:, 1] #Unpacking the data on the contour.
    angular_frequency_uncertainty = np.abs(np.max(
        angular_frequencies - np.min(angular_frequencies))) / 2
    velocity_uncertainty = np.abs(np.max(velocities) - np.min(velocities)) / 2
    axis.errorbar(angular_frequency_years, velocity_minimum, fmt='ro',
                  label='Minimum ' r'$\chi^2$ Location')
    axis.errorbar(angular_frequency_years + angular_frequency_uncertainty,
                  velocity_minimum, fmt='go',
                  label='Uncertainty in Angular Frequency')
    axis.errorbar(angular_frequency_years - angular_frequency_uncertainty,
                  velocity_minimum, fmt='go')
    axis.errorbar(angular_frequency_years,
                  velocity_minimum + velocity_uncertainty, fmt='bo',
                  label='Uncertainty in Velocity')
    axis.errorbar(angular_frequency_years,
                  velocity_minimum - velocity_uncertainty, fmt='bo')
    axis.set_title(r'$\chi^2$ Contours Against Parameters.')
    axis.set_ylabel('Velocity $[m/s]$')
    axis.set_xlabel('Angular Frequency $[rad/yr]$')
    axis.collections[0].set_label('1 \u03C3 Uncertainty Contour')
    axis.legend()
    plt.savefig('contour_plot.png', dpi=300)
    plt.show()

    return (velocity_uncertainty, angular_frequency_uncertainty)

def calculate_output_values(velocity_minimum, angular_frequency_minimum,
                            reduced_chi_squared, velocity_uncertainty,
                            angular_frequency_uncertainty):
    """
    Calculates the orbital distance, velocity of the planet and mass of the
    planet while propagating the relevant uncertainties. Also instigates the
    function which offers to write this data to a text file.

    Parameters
    ----------
    velocity_minimum : float
        Velocity of the star which minimises the chi-squared parameter.
    angular_frequency_minimum : float
        Angular frequency which minimises the chi-squared parameter.
    reduced_chi_squared : float
        Reduced chi-squared value to be outputted to user.
    velocity_uncertainty : float
        Uncertainty in the star's velocity.
    angular_frequency_uncertainty : float
        Uncertainty in the systems angular velocity.

    Returns
    -------
    None.

    """
    orbit_distance = orbital_distance(angular_frequency_minimum) #Distance (AU)
    velocity_planet = planet_velocity(orbit_distance * (1 / METRES_TO_AU)) #Velocity (m/s)
    planet_mass = mass_planet(velocity_minimum, velocity_planet) #Mass (Jovians)
    orbital_distance_uncertainty = orbit_distance_uncertainty(
        angular_frequency_uncertainty, angular_frequency_minimum)
    velocity_planet_uncertainty = planet_velocity_uncertainty(
        orbital_distance_uncertainty * (1 / METRES_TO_AU),
        orbit_distance * (1 / METRES_TO_AU))
    planet_mass_unc = mass_planet_uncertainty(velocity_minimum, velocity_planet,
                                              velocity_uncertainty,
                                              velocity_planet_uncertainty,
                                              planet_mass * (1 / KG_TO_JOVIAN))
    print('\n''Velocity of star: ({:2.2f} ' u'\u00B1 {:2.2f}) m/s. \n'
          'Angular frequency of motion: ({:.4} ' u'\u00B1 {:.2}) rad/s. \n'
          'Mass of Planet: ({:1.3f} ' u'\u00B1 {:0.3f}) Jovian Masses. \n'
          'Orbital distance of planet: ({:1.3f} ' u'\u00B1 {:0.3f}) AU. \n'
          'Reduced chi-squared value: {:0.3f}. \n'
          'Velocity of planet: ({:4.0f} ' u'\u00B1 {:2.0f}) m/s.'
          .format(velocity_minimum, velocity_uncertainty,
                  angular_frequency_minimum, angular_frequency_uncertainty,
                  planet_mass, planet_mass_unc, orbit_distance,
                  orbital_distance_uncertainty, reduced_chi_squared,
                  velocity_planet, velocity_planet_uncertainty))
    data = [velocity_minimum, velocity_uncertainty, angular_frequency_minimum,
            angular_frequency_uncertainty, planet_mass, planet_mass_unc,
            orbit_distance, orbital_distance_uncertainty,
            reduced_chi_squared, velocity_planet, velocity_planet_uncertainty]
    write_to_file(data)

def write_to_file(data):
    """
    Offers to write the calculated data to a text file.

    Parameters
    ----------
    data : list
        A list holding all of the key data for the star-planet system.

    Returns
    -------
    None.

    """
    file_write = input(
        "Would you like the data writing to a text file? (y/n) ")
    if file_write == 'y':
        [velocity_minimum, velocity_uncertainty, angular_frequency_minimum,
         angular_frequency_uncertainty, planet_mass, planet_mass_unc,
         orbit_distance, orbital_distance_uncertainty,
         reduced_chi_squared, velocity_planet, velocity_planet_uncertainty] = data
        file_object = open("spectroscopy_data.txt", 'a')
        file_object.write('Velocity of star: ({:2.2f} ' u'\u00B1 {:2.2f}) m/s. '
                          'Angular frequency of motion: ({:1.3e} ' u'\u00B1 '
                          '{:1.1e}) rad/s.' '\n' 'Mass of Planet: '
                          '({:1.3f} ' u'\u00B1 {:0.3f}) Jovian Masses. Orbital '
                          'distance of planet: ({:1.3f} ' u'\u00B1 {:0.3f}) AU.' '\n'
                          'Reduced chi-squared value: {:.3f}. Velocity of planet: '
                          '({:4.0f} ' u'\u00B1 {:2.0f}) m/s. \n'.format(
                              velocity_minimum, velocity_uncertainty,
                              angular_frequency_minimum,
                              angular_frequency_uncertainty, planet_mass,
                              planet_mass_unc, orbit_distance,
                              orbital_distance_uncertainty, reduced_chi_squared,
                              velocity_planet, velocity_planet_uncertainty))
        file_object.close()
        print("The calculated data was recorded in spectroscopy_data.txt")

def time_of_orbit(angular_frequency_minimum):
    """
    Displays a loading bar and allows the user to get a sense of the length of
    time for an orbit in the planet-star system.

    Parameters
    ----------
    angular_frequency_minimum : float
        The angular frequency which minimises the chi squared parameter.

    Returns
    -------
    None.

    """
    run_loop = True
    while run_loop is True:
        want_time = input(
            "Would you like a rough sense of how long this orbit lasts? (y/n) "
            ).lower()
        time_elapse = ((
            np.pi * 2) / angular_frequency_minimum) * SECONDS_TO_YEARS
        if want_time == 'y':
            print('Each second which elapses here is worth 1 '
                  'year in our distant galaxy.')
            pause(4)
            print('Orbit starts now:')
            pause(0.5)
            multiplied_rounded_time = int(round(time_elapse, 1)) * 10
            for dummy_time in counter(range(multiplied_rounded_time),
                                      desc='Percent of total orbit',
                                      total=multiplied_rounded_time):
                pause(0.1)
            run_loop = False
        elif want_time == 'n':
            run_loop = False
        else:
            print('Please enter either y or n.')

#-----Main Code-----

if check_data_files(FILE_1, FILE_2) is True:
    TIMES, VELOCITIES, VELOCITY_UNCERTAINTIES = reading_and_preparing_data(
        FILE_1, FILE_2, INCLINATION)

    [TIMES, VELOCITIES, VELOCITY_UNCERTAINTIES, VELOCITY_MINIMUM,
     ANGULAR_FREQUENCY_MINIMUM] = fit_with_outliers(TIMES, VELOCITIES,
                                                    VELOCITY_UNCERTAINTIES)

    VELOCITY_MINIMUM, ANGULAR_FREQ_MINIMUM, REDUCED_CHI_SQUARED = second_fit_plot(
        VELOCITY_MINIMUM, ANGULAR_FREQUENCY_MINIMUM, TIMES, VELOCITIES,
        VELOCITY_UNCERTAINTIES)

    VELOCITY_UNCERTAINTY, ANGULAR_UNCERTAINTY = contour_plot_and_uncertainties(
        ANGULAR_FREQ_MINIMUM, VELOCITY_MINIMUM,
        REDUCED_CHI_SQUARED * (len(TIMES) - 2), TIMES)

    calculate_output_values(VELOCITY_MINIMUM,
                            ANGULAR_FREQ_MINIMUM * SECONDS_TO_YEARS,
                            REDUCED_CHI_SQUARED, VELOCITY_UNCERTAINTY,
                            ANGULAR_UNCERTAINTY * SECONDS_TO_YEARS)
    time_of_orbit(ANGULAR_FREQ_MINIMUM * SECONDS_TO_YEARS)
