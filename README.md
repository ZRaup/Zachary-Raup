 ![](port_5x4ss.png)  

# Zachary Raup

[Link to CV:](CV_Raup_Z.pdf)

<a href = "https://www.linkedin.com/in/zachary-raup-6280a3265"><img src="https://img.shields.io/badge/-LinkedIn-0072b1?&style=for-the-badge&logo=linkedin&logoColor=white" /></a>

### About Me
Welcome to my data science portfolio. I graduated Summa Cum Laude from Kutztown University with a bachelor's degree in Physics, driven by an insatiable curiosity and a passion for data-driven exploration. Throughout my academic journey, I specialized in data analysis and modeling, particularly in the realms of exoplanets and binary stars. This experience not only deepened my understanding of complex datasets but also honed my skills in extracting meaningful insights.

Professionally, I have developed robust capabilities in Python for data analysis, leveraging techniques such as exploratory data analysis, data cleaning, and statistical modeling. I am proficient in SQL querying, adept at managing and extracting insights from large datasets to support data-driven decision-making. Additionally, I possess advanced skills in data visualization using Tableau and Power BI, creating insightful visualizations that effectively communicate findings and support strategic initiatives.

I am dedicated to applying my expertise in Python, SQL, and data visualization tools to solve challenging problems and contribute meaningfully to data-driven projects. I thrive in collaborative environments and am committed to continuous learning and professional growth in the dynamic field of data science.

&nbsp;  

Below is a select list of data and coding projects that I have completed:

## [Project 1: Utilizing MCMC in Python to Explore the Parameter Space of an Exoplanet Transit](TOI4153_port.ipynb)

### Objective: 
##### This research project aims to model the transit of an exoplanet across a star using the Python package 'batman'. The objective is to predict changes in the star's brightness during these transits and assess the accuracy of the model against observed photometry data from the CR Chambliss Astronomical Observatory.

### Methodology:
##### The methodology involves developing a physics-based model using 'batman' to simulate exoplanet transits. A log likelihood function is employed to quantify the model's fit to observational data, utilizing photometry data collected from the observatory. The Markov Chain Monte Carlo (MCMC) algorithm, implemented via 'emcee', is utilized to sample from the posterior distribution of model parameters. This approach allows exploration of parameter space and estimation of uncertainties associated with key parameters such as planet radius, impact parameter, and time of mid-transit.

### Data: 
##### The primary data source for this study is photometry data obtained from the CR Chambliss Astronomical Observatory. These observational data points are crucial for refining the exoplanet transit model and validating its accuracy against real-world observations.

### Visualization: 
##### Visualization of the research findings is achieved through the use of the Python package matplotlib. This includes generating light curves that illustrate how the star's brightness changes over time during exoplanet transits. Additionally, histograms are used to visualize the probability distributions of model parameters, providing insights into uncertainties and likelihoods associated with each parameter estimation. A corner plot is employed to depict correlations and interactions between different model parameters, offering a comprehensive view of the model's complexity and relationships.

### Presentation: 
##### The presentation of research findings involves clear and informative visualizations that effectively communicate the quality of the model fit and the implications of the study's results. These visual aids not only highlight quantitative insights into parameter uncertainties but also validate the model against observational data from the CR Chambliss Astronomical Observatory. Additionally, I had the opportunity to present this research project at the 241st AAS (American Astronomical Society) meeting. The goal is to provide a robust analysis of exoplanet transit dynamics, contributing to our understanding of planetary systems beyond our solar system.

Key Components
Exoplanet Transit Modeling:
Using the Python package 'batman', I developed a model to simulate the transit of an exoplanet across a star. This model employs physics-based equations to predict how the brightness of the star changes as the exoplanet passes in front of it, known as the transit.

Log Likelihood Function:
The log likelihood function quantifies how well the exoplanet transit model fits observed photometry data collected from the CR Chambliss Astronomical Observatory. This function is crucial for assessing the statistical likelihood of different parameter values within the model.

MCMC Algorithm:
I utilized the Python package 'emcee' to implement the Markov Chain Monte Carlo (MCMC) algorithm, which samples from the posterior distribution of model parameters given the photometry data. MCMC is particularly useful for exploring parameter space and quantifying uncertainties in measurements.

Data Source:
I collected the photometry data used in this analysis from the CR Chambliss Astronomical Observatory on Auguest 22202, 2202, using a blue and infrared filter. These observational data points are essential for refining the exoplanet transit model and validating its accuracy against real-world observations.

Visualization:
To visualize the results of the exoplanet transit model, I utilized the Python package matplotlib to generate several types of plots:

- Light Curves: These plots illustrate how the brightness of the star changes over time during the exoplanet transit, providing a direct representation of the modeled data against observed photometry.

- Parameter Probability Distributions: Histograms were employed to display the probability distributions of each model parameter (such as planet radius, impact parameter, and time of mid-transit). These distributions offer insights into the uncertainties and likelihoods associated with each parameter estimation.

- Corner Plot: A corner plot was used to visualize the correlations and parameter space relationships between different model parameters. This type of plot aids in understanding how changes in one parameter affect others, offering a comprehensive view of the model's complexity.

These visualizations play a crucial role in assessing the quality of the model fit and effectively communicating the findings of the study. They provide both quantitative insights into parameter uncertainties and qualitative validation against observational data from the CR Chambliss Astronomical Observatory.



#### Image 1:
##### Light curve of TOI 5143 data (CRCAO) and model (batman)
 ![](lightkurve.png)

#### Image 2:
###### Corner plot of parameter space and histogram for the parameters of Planet Radius (Rp), Impact Parameter (b), and Mid-Transit Time (t0)
 ![](cornerplot.png)  

