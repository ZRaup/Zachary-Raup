# Zachary Raup

[Link to CV:](CV_Raup_Z.pdf)

<a href = "https://www.linkedin.com/in/zachary-raup-6280a3265"><img src="https://img.shields.io/badge/-LinkedIn-0072b1?&style=for-the-badge&logo=linkedin&logoColor=white" /></a>

### About Me
sjakjns jsnka


Below is a select list of data and coding projects that I have completed:

## [Project 1: Utilizing MCMC in Python to Explore the Parameter Space of an Exoplanet](TOI4153.ipynb)

#### Key Components
##### Exoplanet Transit Model: 
Python package, batman,You've mentioned creating a model of an exoplanet transit. This model likely uses physics-based equations to predict how the brightness of the star changes as the exoplanet transits in front of it.

##### Log Likelihood Function: 
This function quantifies how well your model (presumably the exoplanet transit model) fits the observed photometry data. The MCMC algorithm uses this to evaluate the likelihood of different parameter values.

##### MCMC Algorithm: 
The python package, emcee, is used to sample from the posterior distribution of your model parameters (like Radius of Planet, Impact Parameter, Time of Mid-Transit) given your data. It's great for exploring parameter space and understanding the uncertainties in your measurements.

##### Data Source: 
Photometry data from the CR Chambliss Astronomical Observatory. This data is crucial for fitting your model and validating your results.

##### Visualization: 
Python package matplotlib. mentioned plotting a light curve of the exoplanet transit based on the photometry data. Visualizing your results helps in understanding the quality of your fit and in communicating your findings.


![](lightkurve.png)   ![](cornerplot.png)
