# Utilizing MCMC in Python to Explore the Parameter Space of an Exoplanet
###### By: Zachary Raup


### Objective:
This research project aims to model the transit of an exoplanet across a star using the Python package 'batman'. The objective is to predict changes in the star's brightness or flux during these transits and assess the accuracy of the model against observed photometry data from the CR Chambliss Astronomical Observatory (CRCAO).   

### Methodology:
The methodology involves developing a physics-based model using the python package'batman' to simulate exoplanet transits. A log likelihood function is employed to quantify the model's fit to observational data, utilizing photometry data I collected from CRCAO. The Markov Chain Monte Carlo (MCMC) algorithm, implemented via package 'emcee', is utilized to sample from the posterior distribution of model parameters. This approach allows exploration of parameter space and estimation of uncertainties associated with key parameters such as planet radius, impact parameter, and time of mid-transit.

### Data: 
The primary data source for this project is photometry data of exoplanet TOI-4153 obtained from the CR Chambliss Astronomical Observatory at Kutztown University in Kutztown, PA. The data was collected on September 16, 2022 in two different filters, blue, and infrared. These observational data points are crucial for refining the exoplanet transit model and validating its accuracy against real-world observations.

### Visualization: 
Visualization of the research findings is achieved through the use of the Python package 'matplotlib'. This includes generating light curves that illustrate how the star's brightness changes over time during exoplanet transits. Additionally, histograms are used to visualize the probability distributions of model parameters, providing insights into uncertainties and likelihoods associated with each parameter estimation. A corner plot is employed to depict correlations and interactions between different model parameters, offering a comprehensive view of the model's complexity and relationships.

### Presentation: 
I had the opportunity to present this research project at the 241st AAS (American Astronomical Society) meeting. The presentation of research findings involves clear and informative visualizations that effectively communicate the quality of the model fit and the implications of the project's results. These visual aids not only highlight quantitative insights into parameter uncertainties but also validate the model against observational data from the CR Chambliss Astronomical Observatory. The goal is to provide a robust analysis of exoplanet transit dynamics, contributing to our understanding of planetary systems beyond our solar system.


```python

```

  
### Key Components  
##### Exoplanet Transit Modeling:
Using the Python package 'batman', I developed a model to simulate the transit of an exoplanet across its host star. This model employs physics-based equations using systematic parameters to predict how the brightness of the star changes as the exoplanet passes in front of the star, from the observes perspective, known as the transit. Exoplanet transits give crucial details about the potential exoplanets size.

##### Log Likelihood Function:
The log-likelihood function is a concept in statistical inference and is used to estimate the parameters of a statistical model. For exoplanet photometry, the log-likelihood function is particularly useful for fitting models to observed data, such as the transit light curve of an exoplanet. The log likelihood function quantifies how well the exoplanet transit model fits observed photometry data collected from the CR Chambliss Astronomical Observatory. This function is crucial for assessing the statistical likelihood of different parameter values within the model.

##### MCMC Algorithm:
I utilized the Python package 'emcee' to implement the Markov Chain Monte Carlo (MCMC) algorithm, which samples from the posterior distribution of model parameters given the photometry data. I used the prior known parameter values from the ExoFOP database (Link in next section) to run the algoristhm. MCMC is particularly useful for exploring parameter space and quantifying uncertainties in measurements.

##### Visualization:
To visualize the results of the exoplanet transit model, I utilized the Python package matplotlib to generate several types of plots:

- Light Curves: These plots illustrate how the brightness of the star changes over time during the exoplanet transit, providing a direct representation of the modeled data against observed photometry.

- Parameter Probability Distributions: Histograms were employed to display the probability distributions of each model parameter (such as planet radius, impact parameter, and time of mid-transit). These distributions offer insights into the uncertainties and likelihoods associated with each parameter estimation.

- Corner Plot: A corner plot was used to visualize the correlations and parameter space relationships between different model parameters. This type of plot aids in understanding how changes in one parameter affect others, offering a comprehensive view of the model's complexity.

These visualizations play a crucial role in assessing the quality of the model fit and effectively communicating the findings of the study. They provide both quantitative insights into parameter uncertainties and qualitative validation against observational data from the CR Chambliss Astronomical Observatory.





```python

```


##  Exoplanet
###  TOI-4153.01 | TIC 470171739 

## TOI-4153 is a yellowish-white k tyoe star that has a temperature of 6411 [(ExoFOP)](https://exofop.ipac.caltech.edu/tess/target.php?id=470171739)

Image 1: TOI-4153 Data Image
dsijiaksm aikms
![I_field.png](attachment:I_field.png)


### Known Parameters taken from ExoFOP:
##### https://exofop.ipac.caltech.edu/tess/target.php?id=470171739




#### Web TESS Viewing Tool
##### https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py?Entry=470171739




```python

```

#### The following is a portion of the Python code used the completion of this project


```python

```


```python
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import batman
import emcee as mc
import corner as corn
```


```python
# Use pandas to import CSV Data file (x, y, err):   
FILE = "C:/Users/zraup/470171739b.csv"
CRCAO_data = pd.read_csv(FILE, delim_whitespace=False, header = 0).to_numpy()
```


```python
# Identify Data

# Ephemeris for TOI 470171739
P = 4.617                 # Period (days)  (ExoFop)
T0 = 2459838.72           # Epoch (BJD)    (From Observation)

# Data from CRCAO observations TJD = data[:,0]       
flux_n = data[:,1]        # normaled fulx 
Error= data[:,2]          # error from data
BJD = TJD + 2457000       # convert TJD to Baryventric Julian Date

# Predicted TTF ingress and egress
ing_TTF = 2459838.6277    # BJD
egr_TTF = 2459838.8136    # BJD
T0_TTF = 2459838.7206 
```


```python
# Log likelyhood funtion Probability
def logp(freeparams, BJD):
   
    # Parameters (Set free parameters(unknown) and fixed(known))           
    rplanetRearth = freeparams[0]       # free     # ExoFop 17.5913          
    rstarRsun = 1.672                              # ExoFop 1.672
    mstarMsun = 1.289                              # ExoFop 1.289
    ainAU = (P/365.25)**(2./3.)*mstarMsun**(1./3.) # Kep 3rd Law
    params = batman.TransitParams()                # exoplanet parameters from batman
    params.t0 = freeparams[1]           # free     # time of inferior conjunction       
    b = freeparams[2]                   # free     # impact parameter 
    params.per = P                                 # orbital period
    params.rp = rplanetRearth/rstarRsun/110.       # planet radius (in units of stellar radii)
    params.a = ainAU*215/rstarRsun                 # semi-major axis (in units of stellar radii)
    params.inc = 180*np.arccos(b/params.a)/np.pi   # orbital inclination (in degrees)
    params.ecc = 0.00                              # eccentricity
    params.w = 90.0                                # longitude of periastron (in degrees)
    params.u = [0.3, 0.3]                          #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       
    
# Set-Up Model using Batman transit modeler
    bat = batman.TransitModel(params, BJD)         # initialize model
    flux_mc = bat.light_curve(params)              # model of flux
    
    # Calculate difference between data collected and model
    dif_mc = data[:,1] - flux_mc
    
# Set restrictions for paramters   
    # Do not allow b to be less than 0 or greater than 1
    if b < 0: 
        return -1*np.inf
    if b > 1:
        return -1*np.inf
    
# Define log like funtion
    loglike = -0.5 * np.sum(dif_mc**2 / data[:,2]**2)

# Do not allow loglike to be (-)
    if loglike == loglike:
        return loglike
    else: 
        return -1*np.inf
    
```


```python
# Set dimensions of sampler 
ndim = 3                                              # number of free params

# Set number of walkers
nwalkers = 40 

# Set the Prior for Rp, t0, and impact parameter b       # initial (Buest Guess) parameter values for walkers
r0 = [16.66, T0, 0.56]                                  # Set initial values for walkers

# Set starting point for each walker and allow them to move randomly
Rp0_walker = r0[0] + 1e-3 * np.random.randn(nwalkers)   # walkers for Radius of Planet
T0_walker = r0[1] + 1e-3 * np.random.randn(nwalkers)    # walkers for T0 #  (BJD)
b0_walker = r0[2] + 1e-3 * np.random.randn(nwalkers)    # waklkers for b

# Transpose 
r0 = np.transpose([Rp0_walker, T0_walker, b0_walker])  

```


```python
# Initialize the Sampler
sampler = mc.EnsembleSampler(nwalkers, ndim, logp, args=[BJD])
```


```python
# Run the Sampler:
state = sampler.run_mcmc(r0, 25000, progress = True)     # sampler.run_mcmc(initial values, number of itierations)
r = sampler.get_chain(flat=True, discard= 2000)          # Discard initial steps of walkers, allow time for convergence
```


```python

```

### Image 2: TOI-4153 lightcurve
CRCAO photomery data of the TOI-4153.01 transit in a blue and infrared filer. Model is made using the batman transit modeler, see Table 1 for parameters of model.
![lightkurve-3.png](attachment:lightkurve-3.png)

### Image 3: emcee Radius Estimation Distribution
The median of the emcee estimation distribution for the radius of TOI-4153.01 was calculated to be 16.65 Earth Radii given the prior of 16.66 Earth Radii from ExoFOP. The distribution shows a strong convergence for the planets' radius based on the CRCAO data and the batman model.
![hist.png](attachment:hist.png)

### Image 4: Posterior Probability Corner Plot
Corner plot showing the posterior probability for the free parameters, Planet Radius, Mid-Transit Time, and Impact Parameter. Plots show the

![cornerplot-2.png](attachment:cornerplot-2.png)

### Table 1: Parameter Table
Data table showing the parameters used for the light curve model. The free parameters are represented by the median emcee distribution approximation values and the fixed data values were taken from the [ExoFOP database](https://exofop.ipac.caltech.edu/tess/target.php?id=470171739) Incldued in the free parameters, Inclination and semi major axis due values being calculated from another free parameter.
![table.png](attachment:table.png)

### Table 2: Radius Comparison
The table shows the modeled radius for three TOI`s using CRCAO data and the batman model, astroimaheJ model and the predicted radius beofre data collection from the TESS Tranisit Finder (TTF).
![table1.png](attachment:table1.png)

### Conclusion:
Using a Markov Chain Monte Carlo algorithm from the 'emcee' Python package that sampled CRCAO photmetry data using a 'batman' transit modeler Python and a log liklifood function. The radius of TOI-4153 was estimated to be 16.65 Earth Radii with a standard deviation of 0.49.


```python
fix objtive, caprions, image 1, data section, almost



estimatuib 16.65 +/- 0.49 68% confidence interval
```


      File "<ipython-input-1-bd36347eac1b>", line 1
        fix objtive
            ^
    SyntaxError: invalid syntax
    



```python

```
