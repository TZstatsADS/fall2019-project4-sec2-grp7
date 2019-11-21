# Project 4: Algorithm implementation and evaluation: Collaborative Filtering

### [Project Description](doc/project4_desc.md)

Term: Fall 2019

+ Team #7 
+ Projec title: Comparison betweeen KNN and Kernel ridge regression with gradient descent with probabilistic assumptions
+ Team members
	+ Sim, Young js5134@columbia.edu
	+ Sohn, Jongyoon js5342@columbia.edu
	+ Gao, Xin xg2298@columbia.edu
	+ Yang, Siyu sy2796@columbia.edu
	+ Meng, Yang ym2696@columbia.edu
+ Project summary: The project estimates latent factors by gradient descent with probabilistic assumptions to create user-factor matrix and item-factor matrix. In post-processing, we used KNN and Kernel Ridge Regression to construct the second regression terms to improve the model, and combined the result with the original model using linear regression. We then campared the two methods.
	
**Contribution statement**: All team members approve our work presented in this GitHub repository including this contributions statement. 
+ Young worked on PMF, KNN, and Kernel Ridge Regression, and the main script in python (PMF_model.py and PMF_main.ipynb).
+ Jongyoon created slides for presentation (Probabilistic Matrix Factorization.pdf) and worked on parameter tuning.
+ Xin worked on PMF in R (Matrix Factorization1.R) and parameter tuning.
+ Siyu worked on KNN in R (knn other version.Rmd) and parameter tuning.
+ Yang worked on KNN in R (knn.Rmd), KNN function in python (knn.r) and parameter tuning.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
