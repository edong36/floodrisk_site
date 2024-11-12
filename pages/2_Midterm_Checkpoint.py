import streamlit as st

st.set_page_config(page_title="Midterm Checkpoint")

st.title("Midterm Checkpoint")

st.header('Introduction and Background')
st.write('Flooding is a destructive force for communities. Because of abundant climate and weather data, it is difficult for humans to interpret this data to predict flooding. Machine learning (ML) provides new ways to utilize this data efficiently. Varying approaches including artificial neural networks, decision trees including random forests, long short-term memory (LSTMs), and support vector machines (SVMs) have been useful in flood prediction [1][2], and the accuracy of these is improved by optimizing the models and combining the models through algorithm ensemble [1].')
st.write('Flood risk is often evaluated based on rainfall and water proximity. One of the datasets we will be using is the Georgia rainfall indicator that catalogs rainfall totals over monthly periods. We expect the anomaly feature will be utilized here (https://data.humdata.org/dataset/geo-rainfall-subnational). We will also utilize FEMA flood insurance information to help determine the location and extent of flooding. The latitude, longitude, and date of loss features will be used (https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2). Note that these may change based on the status of the NCEI Archive which houses varying rainfall and flood disaster data sets and is currently inaccessible due to damage from Hurricane Helene.')

st.header('Problem Definition')
st.write('To address the need for flooding predictors, we will first compile different types of weather and climate data. Then, we will reduce the compiled data\'s dimensionality through principal component analysis (PCA) and finally, normalize the data to improve readability and clarity. We plan to analyze the data using LSTMs, SVMs, and random forest models. These each present their own unique advantages.')

st.header('Methods')
st.write('To begin the preprocessing of our dataset, we removed unwanted features [2] and cleaned our data by filling in missing data points via interpolation. We then utilized principal component analysis (PCA), a dimensionality reduction technique effective in simplifying datasets by reducing the number of features while still maintaining and maximizing variance in the data to make patterns more discernible before applying a machine learning algorithm [4]. This also helps with preventing overfitting. Lastly, we normalized to ensure that no single feature dominates the model due to scale differences [5].')
st.write('For our machine learning algorithm, we implemented long short-term memory (LSTM), a type of recurrent neural network effective for capturing sequential long-term dependencies in time series data[2]. Our dataset is a great example of this as it captures several predictors of flooding over a long period of time such as base flood elevation, lowest floor elevation, cause of damage, and precipitation. The LSTM model was tuned for different time steps and model depths to identify optimal configurations, aiming to capture temporal patterns effectively.')

st.header('Results and Discussion')
st.write('After running LSTM on our dataset, we achieved very high accuracy. In the classification report below, class 0 represents a “No Flood” prediction and class 1 represents a “Flood” prediction. As can be seen, the model predicted these events with 97&percnt; and 99&percnt; accuracy, respectively.')
st.image('images/classification report.png')
st.write('We also produced a confusion matrix which shows that the model performs very well in predicting both events, successfully identifying “No Flood” events 96.93&percnt; of the time and “Flood” events 99.49&percnt; of the time. It only reported “Flood” events incorrectly as “No Flood” events 0.51&percnt; of the time and “No Flood” events as “Flood” events incorrectly 3.07&percnt; of the time. Overall, this model is very reliable in classifying between “No Flood” and “Flood” events.')
st.image('images/confusion matrix.png')
st.write('The following decision tree model explains how the classification between "No Flood" and "Flood" events is made. At each decision point, the model splits the data based on specific feature values to separate the two classes, with each group classified according to whether it contains more "No Flood" or "Flood" events. This process continues until the final classification at the leaf nodes, which represents the maximum depth that we set to prevent overfitting.')
st.image('images/decision tree.png')
st.write('We also performed tuning analysis to determine the ideal number of time steps and depth to produce the best results. We measured mean loss, mean accuracy, and mean F1 score along with standard deviations of these metrics and found that somewhere between 10-15 time steps produced the best results.  This, combined with good data preprocessing helped us achieve great results.')
st.image('images/tuning results.png')
st.image('images/tuning visualization.png')
st.write('Overall, the LSTM model proved to be a highly effective solution for flood prediction in this context. In the coming weeks, we plan to explore the viability of support vector machines and random forest models in predicting flood events. In addition to these models that we\'ll explore, we\'ll also analyze how we can produce the best model for predicting flood events.')

st.header('References')
st.write('[1] A. Mosavi, P. Ozturk, and K. Chau, "Flood Prediction Using Machine Learning Models: Literature Review," Water, vol. 10, no. 11, p. 1536, Oct. 2018, doi: https://doi.org/10.3390/w10111536.')
st.write('[2] X.-H. Le, H. V. Ho, G. Lee, and S. Jung, “Application of Long Short-Term Memory (LSTM) Neural Network for Flood Forecasting,” Water, vol. 11, no. 7, p. 1387, Jul. 2019, doi: https://doi.org/10.3390/w11071387.')
st.write('[3] The Click Reader, “Data Preprocessing in Python — Handling Missing Data,” Medium, Sep. 21, 2021. https://medium.com/@theclickreader/data-preprocessing-in-python-handling-missing-data-b717bcd4a264 (accessed Oct. 04, 2024).')
st.write('[4] J. Murel, "What is Dimensionality Reduction? | IBM," www.ibm.com, Jan. 05, 2024. https://www.ibm.com/topics/dimensionality-reduction')
st.write('[5] A. Bhandari, “Feature Scaling | Standardization Vs Normalization,” Analytics Vidhya, Apr. 03, 2020. https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/')
st.write('[6] S. Nevo et al., “Flood forecasting with machine learning models in an operational framework,” Hydrology and Earth System Sciences, vol. 26, no. 15, pp. 4013-4032, Aug. 2022, doi: https://doi.org/10.5194/hess-26-4013-2022.')
st.write('[7] M. Khan, Afed Ullah Khan, B. Ullah, and S. Khan, “Developing a machine learning-based flood risk prediction model for the Indus Basin in Pakistan,” Water Practice & Technology, vol. 19, no. 6, pp. 2213-2225, Jun. 2024, doi: https://doi.org/10.2166/wpt.2024.151.')
st.write('[8] A. Bajaj, “Performance Metrics in Machine Learning [Complete Guide],” neptune.ai, May 02, 2021. https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide')

st.header('Gantt Chart')
st.write('https://docs.google.com/spreadsheets/d/14JV9ATnk5HLtbtUmybd7S8qw2GmBIz_9/edit?usp=sharing&ouid=108468648048564250512&rtpof=true&sd=true')
st.header('Contribution Table')
st.write('https://docs.google.com/spreadsheets/d/1dApQIcvUl-MthNjfa1V3ELUzFv5_YK5NvMCyhfdKis0/edit?gid=0#gid=0')