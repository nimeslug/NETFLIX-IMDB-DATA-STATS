# NETFLIX IMDB SCORES - DATA SCIENCE ANALYSIS

This repository contains a data science project focused on analyzing Netflix movie and TV show data, specifically IMDB scores, using various statistical methods.

## Project Overview

In this project, different statistical techniques such as Z-tests, T-tests, Spearman correlation, and Shapiro-Wilk test are applied to a Netflix dataset to explore various relationships between movie and TV show ratings. The dataset contains information on Netflix's content, including IMDB scores, release years, movie types, age certifications, and more.

## Key Steps

1. **Data Cleaning**:
   
   - Removed unnecessary columns and handled missing values.
     
   - Explored the dataset and identified statistical properties.

3. **Statistical Tests**:
   
   - Conducted Z-test and T-test to compare sample means against population parameters.
     
   - Applied the Shapiro-Wilk test to assess normality of the data.
     
   - Performed the Anderson-Darling test for normality on release years.
     
   - Used the Spearman correlation test to examine the relationship between IMDb votes and movie runtimes.
     
   - Conducted Mann-Whitney U test for comparing IMDb votes and runtime.

4. **Data Visualization**:
   
   - Created various plots to visualize distributions, trends, and correlations in the data, including:
     
   - Pie chart for movie type distribution.
       
   - Bar plot for top 10 rated movies.
       
   - Histograms and boxplots for various features like release year, IMDb votes, and runtime.

5. **Hypothesis Testing**:

   Conducted hypothesis tests to check if the sample means are significantly different from population means, especially focusing on IMDb scores and release years.

## Requirements

To run this project, you will need the following Python packages:

- pandas
 
- numpy

- matplotlib
  
- seaborn
  
- scipy



