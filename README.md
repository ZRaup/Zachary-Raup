![](self158.png)

# Zachary Raup
--- 
<a href = "https://www.linkedin.com/in/zachary-raup-6280a3265"><img src="https://img.shields.io/badge/-LinkedIn-0072b1?&style=for-the-badge&logo=linkedin&logoColor=white" /></a>  
[Zachary's Resume (pdf)](Res_Raup_Z.pdf)  
[Data Science Courses Completed](courses.md) 

### About Me
Welcome to my data science portfolio! I'm **Zachary Raup**, a data scientist with a strong foundation in physics and a passion for uncovering insights from complex datasets. I graduated Summa Cum Laude from Kutztown University with a B.S. in Physics, where I focused on data modeling in astrophysical systems—particularly exoplanets and binary stars. This research experience trained me to approach problems analytically, work with real-world uncertainty, and extract meaning from noisy data.

To strengthen my data science skillset, I earned certifications from **DataCamp** in Data Science, Data Analysis, **Python**, and **SQL**, and completed coursework in **machine learning, data preprocessing**, and **visualization**. I apply these skills using Python (pandas, scikit-learn, matplotlib, numpy) and SQL, with additional proficiency in **Tableau** and **Power BI** for data storytelling.

My work focuses on building interpretable, performance-driven models to support real-world decision-making. I enjoy collaborating across disciplines, turning messy data into actionable insight, and constantly learning new techniques to grow as a scientist and developer.

Thanks for visiting—feel free to explore my projects!

&nbsp;  

---

#### Certifications 
- [Data Scientist Associate](DataScienceCert.pdf)
- [Data Analyst Associate](DataAnalystCertificate.pdf)
- [Python Data Associate](PythonDataAssociateCert.pdf)
- [SQL Associate](SQL_cert.pdf)

---  

##### Below is a list of data and coding projects that I have completed. Use project title links to view more information about each project.  
&nbsp;  


### [Project 1](deep-learning-chest-x-ray-pneumonia-detection.ipynb) 
## [Chest X-Ray Pneumonia Detection with Deep Learning](deep-learning-chest-x-ray-pneumonia-detection.ipynb)  

#### Project Overview
Developed a deep learning pipeline to classify chest X-rays as Normal or Pneumonia using an ensemble of pretrained CNNs (ResNet18, DenseNet121, EfficientNet-B0). Achieved a 91.2% test accuracy and an F1-score of 0.9332, with all models demonstrating high pneumonia recall, minimizing false negatives.

*Key components*:
- Implemented 3-fold stratified cross-validation and early stopping to ensure robust model performance
- Applied class-weighted loss functions to address class imbalance during training
- Integrated Grad-CAM visualizations to highlight key diagnostic regions and enhance interpretability for clinical use

This project showcases how deep learning and explainable AI can support radiologists by improving diagnostic accuracy and transparency in medical imaging.

###### Skills Applied: PyTorch, Convolutional Neural Networks (CNN), Computer Vision, Deep Learning, Medical Imaging, Scikit-Learn and more

#### Figure 1: Grad-CAM Heatmap: Interpretable Pneumonia Detection from Chest X-Ray
##### Grad-CAM visualization for a test chest X-ray correctly classified as Pneumonia by the ResNet18 model. The highlighted activation regions (in red and yellow) suggest the model focuses on areas of increased radiographic opacity within the central and lower lung fields—features often indicative of pulmonary infection. This supports the model's decision-making process and provides interpretability in a clinical context.

<img src="GradCam.png" width="350" />  

#### Figure 2: Confusion Matrix: ResNet18 Model on Chest X-Ray Test Set
##### Confusion matrix for the ResNet18 model (fold 0) evaluated on the test set. The model correctly identified 384 of 390 Pneumonia cases (high sensitivity) and 185 of 234 Normal cases, resulting in 6 false negatives and 49 false positives. This performance highlights the model’s strong bias toward minimizing missed Pneumonia diagnoses, a clinically preferred trade-off in high-stakes triage settings.

<img src="confus_matr.png" width="470" />  

 &nbsp;   



### [Project 2](https://www.kaggle.com/code/zacharyraup/discovering-similar-songs)    
## [Discovering Similar Songs Using Machine Learning | Unsupervised Learning with Spotify Data](https://www.kaggle.com/code/zacharyraup/discovering-similar-songs)  

#### Project Overview
This project applies unsupervised machine learning techniques to uncover patterns in Spotify audio data and recommend musically similar songs. By using Non-negative Matrix Factorization (NMF) for dimensionality reduction and t-distributed Stochastic Neighbor Embedding (t-SNE) for visualization, the feature space of over 6,000 tracks was mapped into an interpretable 2D projection. Cosine similarity was then used to identify songs most similar to “Blinding Lights” by The Weekend. The final result is an insightful visual and analytical exploration of musical relationships based on audio characteristics..

###### Skills Applied: Unsupervised Learning, NMF, t-SNE, Cosine Similarity, Data Preprocessing, Python (scikit-learn, NumPy, pandas)

#### Figure 3: Interactive Projection of Songs Colored by Similarity to Blinding Lights - The Weekend
##### This 2D visualization presents a t-SNE projection of the song dataset, with each point representing a track. Colors indicate cosine similarity to “Blinding Lights” by The Weeknd—green represents higher similarity, red represents lower. The 10 most similar songs are marked with square outlines, clearly highlighting clusters of tracks that share sonic traits. This approach visually demonstrates how audio features can be leveraged to identify stylistic similarity between songs.

<img src="tSNE_Blinding_Lights.png" width="900" height="500" />

#### Table 1: Top 10 Most Similar Songs to “Blinding Lights””
##### The table below lists the 10 songs most similar to “Blinding Lights”, based on cosine similarity of audio features. These tracks span multiple genres and artists, yet share core musical qualities such as synth-driven production, emotional tone, and modern pop energy. The diversity of artists—from Post Malone to K/DA—illustrates the reach of “Blinding Lights’” sonic profile.

 **Top 10 Similar Songs to: Blinding Lights - The Weekend**
- Blinding Lights (with ROSALÍA) — Remix - The Weekend, ROSALÍA (Similarity: 0.9988)
- I am not a woman, I'm a god — Halsey (Similarity: 0.9971)
- Feels Great (feat. Fetty Wap & CVBZ) — Cheat Codes, Fetty Wap, CVBZ (Similarity: 0.9951)
- Enough Is Enough — Post Malone (Similarity: 0.9939)
- MI FAI IMPAZZIRE — BLANCO, Sfera Ebbasta (Similarity: 0.9936)
- Last Day Alive — The Chainsmokers, Florida Georgia Line (Similarity: 0.9933)
- POP/STARS — K/DA, Madison Beer, (G)I-DLE, Jaira Burns, League of Legends (Similarity: 0.9931)
- Love It If We Made It — The 1975 (Similarity: 0.9929)
- When I’m Alone — Post Malone (Similarity: 0.9928)
- Together — Sia (Similarity: 0.9928)

 &nbsp;   

### [Project 3](Walmart_Sales_Prediction.md) 
## [Walmart Sales Prediction | Regression Modeling](Walmart_Sales_Prediction.md)  

#### Project Overview
Accurate weekly sales predictions are essential for retail businesses to manage inventory, forecast demand, and optimize profitability. This project explores the use of machine learning techniques to predict weekly sales for Walmart stores based on historical data spanning 2010 to 2012. Various regression models, including Random Forest, Boosted Trees, and Ridge Regression, were applied and compared to identify the most reliable approach for capturing complex data relationships and improving predictive accuracy.

###### Skills Applied: Machine Learning, Python (scikit-learn), Regression Modeling, Data Cleaning, Feature Engineering and more 

#### Figure 4: Average Weekly Sales by Store and Regression Model Performance
##### The plot visualizes the average weekly sales across all stores, revealing that stores like Store 4 and Store 20 consistently outperform others in sales volume, while stores such as Store 33 report the lowest averages.

<img src="avg_sales_store.png" width="350" /> 

#### Table 2: Regression Model Performance
##### The table ranks the performance of various regression models based on RMSE. Random Forest Regression stands out with the lowest RMSE (107,130.99) and highest R² score (0.9636), demonstrating strong predictive accuracy. Decision Tree and Boosted Tree models also show solid performance, whereas linear and neural network models lag behind, highlighting the effectiveness of ensemble methods for this task.

<img src="reg_res.png" width="470" />  


 &nbsp;   

### [Project 4](PDML.md)  
## [Predicting Diabetes Using Machine Learning | Comparison of Classification Models](PDML.md)  

#### Project Overview
This project explores the effectiveness of five machine learning models—Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Machine (SVM)—in predicting diabetes status using a cleaned patient dataset. By employing cross-validation and assessing key metrics such as accuracy, precision, recall, and F1 score, the analysis highlights the importance of selecting a model that balances these metrics for reliable healthcare applications. A model with high accuracy and recall is crucial for effectively identifying diabetic patients, thereby minimizing the risks associated with missed diagnoses. 

###### Skills Applied: Machine Learning, Supervised Learning, Python (scikit-learn), Cross-Validation, Hyperparameter Tuning and more 

#### Figure 5: Classification Model Comparison
##### This boxplot illustrates the cross-validation accuracy of five classification models—Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Machine (SVM). Each box represents the distribution of accuracy scores obtained through 5-fold cross-validation, highlighting the performance stability and variability of each model. The results emphasize the importance of model selection in achieving high accuracy for diabetes classification, crucial for effective healthcare decision-making.

<img src="mod_comp.png" width="500" />  


 &nbsp;   

### [Project 5](TOI4153_port.md) 
## [Utilizing MCMC to Explore the Parameter Space of an Exoplanet Transit](TOI4153_port.md)  

#### Project Overview
This research project focuses on modeling the transit of exoplanets across stars using the Python package 'batman'. The objective was to accurately predict changes in stellar brightness during these transits, validated against photometry data from the CR Chambliss Astronomical Observatory (CRCAO). Methodologically, a physics-based model was developed and evaluated using a log likelihood function to fit observational data. The Markov Chain Monte Carlo (MCMC) algorithm, facilitated by 'emcee', enabled exploration of parameter uncertainties such as planet radius and transit timing. Visualizations created with matplotlib included light curves, histograms of parameter distributions, and a corner plot illustrating parameter correlations. Presenting findings at the 241st AAS meeting highlighted contributions to understanding exoplanet transit dynamics, crucial for advancing knowledge of planetary systems beyond our solar system.

###### Skills Applied: Python (pandas, matplotlib, numpy, emcee, & batman), Jupyter Notebook, and Excel  

#### Figure 6: TOI-4153 Modeled Lightcurve
##### Light curve of TOI-4153 data (CRCAO) taken in a Blue (B) and Infrared (I) filter. The model is built using the Python transit modeler package 'batman'. The parameters of the model were determined using the Markov Chain Monte Carlo algorithm and known parameters taken from the [ExoFOP](https://exofop.ipac.caltech.edu/tess/target.php?id=470171739) database.

 <img src="lightkurve.png" width="500" />  
 

 &nbsp;   
   
### [Project 6](MySQL_port.md) 
## [Insights into Dog Behavior: Analyzing Dognition Data with MySQL](MySQL_port.md)  

#### Project Overview
The goal of this project is to utilize MySQL queries to perform analysis of trends and relationships embedded within the Dognition database. Developed as a fundamental component of the ‘Managing Big Data with MySQL’ course from Duke University, the project focuses on refining and applying skills in data cleaning, sorting, and employing advanced analytical techniques using SQL. By exploring large datasets such as the Dognition database, the project aims to uncover meaningful insights into canine behavior patterns and preferences, leveraging robust data management practices to extract actionable intelligence for further research and practical applications in understanding and enhancing dog-human interactions.

###### Skills Applied: MySQL, Writing Queries, Data Cleaning, and Big Data  

#### Figure 7: Top States by Number of Dognition Users
##### This MySQL query analyzes geographical user trends within the Dognition dataset, resulting in a table that lists the top 5 states with the highest number of Dognition users.

 <img src="state_users.png" width="720" />  



 &nbsp;   

### [Project 7](MuesTabl.md) 
## [Interactive Animation of Museum Visitor Paths and Hourly Room Traffic in Tableau](MuesTabl.md)  

#### Project Overview
The project was undertaken as part of the 'Data Visualization in Tableau' course in Data Camp, where I applied advanced data visualization techniques to transform raw museum data into a meaningful and engaging interactive animation. By leveraging Tableau's powerful features, I was able to create a comprehensive and user-friendly tool that highlights key patterns and trends in museum visitor behavior by the hour. This project not only demonstrates my proficiency in using Tableau for data visualization but also underscores the practical application of these skills in real-world scenarios.

###### Skills Applied: Tableau, Data Visualization

#### Figure 8: Common Musuem Visitor Paths
##### The following three screenshots highlight two popular paths that museum visitors walk at 11 AM, 3 PM, and 6 PM. These images illustrate how the average number of people in each area changes throughout the day, providing insights into visitor flow and behavior

<img src="anim.png" width="700" />  


 &nbsp;   


---
---  

###### Portfolio Links
[Zachary's Portfolio](README.md)  
[Project 1: Chest X-Ray Pneumonia Detection with Deep Learning](deep-learning-chest-x-ray-pneumonia-detection.ipynb)  
[Project 2: Discovering Similar Songs using Machine Learning and Spotify](https://www.kaggle.com/code/zacharyraup/discovering-similar-songs)  
[Project 3: Regression Modeling | Walmart Sales Prediction](Walmart_Sales_Prediction.md)  
[Project 4: Predicting Diabetes Using Machine Learning | Comparison of Classification Models](PDML.md)  
[Project 5: Utilizing MCMC in Python to Explore the Parameter Space of an Exoplanet Transit](TOI4153_port.md)  
[Project 6: Insights into Dog Behavior: Analyzing Dognition Data with MySQL](MySQL_port.md)  
[Project 7: Interactive Animation of Museum Visitor Paths and Hourly Room Traffic in Tableau](MuesTabl.md)  





