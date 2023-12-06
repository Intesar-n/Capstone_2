import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()

# st.set_page_config(page_title= "Retail Platform",page_icon='logo.png')


if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False
    

# Define the welcome page form
if not st.session_state['submitted']:
    
        # st.write("Welcome to Eng.Majed AutoMobile Shop! Please submit to continue.")

        st.markdown("----", unsafe_allow_html=True)
        
        st.title('Retaill store Platform')
        st.subheader("The PERFECT place to derive insights from your data with the magical touch of machine learning")
        st.markdown("----", unsafe_allow_html=True)
        submitted = st.button("Let's GO")
        if submitted:
            st.session_state['submitted'] = True
            st.experimental_rerun()  # Rerun the app to update the state

# Define your tabs
if st.session_state['submitted']:
    tab1, tab2,tab3,tab4 = st.tabs([":blue[Clusters]", ":blue[RFM Details]",":blue[Visualization]",":blue[Details]"])

    with tab4:
        st.markdown('''
       # RFM Customer Segmentation & Cohort Analysis Project

## Introduction
Welcome to the "RFM Customer Segmentation & Cohort Analysis Project", part of a Capstone Project Series designed to enhance skills in data analysis, customer segmentation, and clustering algorithms.

This project focuses on RFM (Recency, Frequency, Monetary) Analysis and its application in customer segmentation, along with data cleaning, data visualization, exploratory data analysis, and cohort analysis. A fundamental knowledge of Python coding and clustering theory is assumed.

## Project Structure
The project is divided into the following main sections:

1. **Data Cleaning & Exploratory Data Analysis**
   - Importing libraries, loading data, and initial data review.
   - Analyzing key variables and customer distribution by country, with a focus on the UK market.

2. **RFM Analysis**
   - Calculating RFM metrics and creating an RFM table for customer segmentation.

3. **Customer Segmentation with RFM Scores**
   - Scoring and categorizing customers based on RFM values.

4. **Applying Clustering**
   - Pre-processing data for clustering.
   - Implementing and comparing different clustering algorithms:
     - K-means Clustering
     - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
     - Gaussian Mixture Model (GMM)
   - Visualizing and interpreting clustering results.

5. **Cohort Analysis**
   - Creating cohorts to analyze customer behavior over time.
   - Tracking key metrics and visualizing results.

## Project Goals
The project aims to provide hands-on experience with:

- RFM Analysis for customer segmentation.
- Data cleaning, visualization, and exploratory analysis.
- Advanced customer segmentation using clustering algorithms.
- Cohort analysis for behavioral tracking.

## Dataset
The Online Retail dataset from the UCI Machine Learning Repository is used, containing transactions for a UK-based online retail between 01/12/2010 and 09/12/2011.

## Tools and Technologies
- Python
- Pandas, Matplotlib, Seaborn
- Scikit-learn for K-means, DBSCAN, and GMM

## How to Use
1. Clone the repository.
2. Install required libraries.
3. Run the Jupyter notebooks sequentially.

## Conclusion
This project provides practical experience in customer segmentation using RFM analysis, various clustering algorithms, and cohort analysis, aiding the understanding of customer behavior and data science applications in marketing.

---

**Author: Eng.MAJED**

**ENGINEER**
''')
   