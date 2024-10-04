import streamlit as st
import pandas as pd

st.title('Flood ML Project Proposal')

st.header('Introduction and Background')
st.write('Flooding is a destructive force for communities. Because of abundant climate and weather data, it is difficult for humans to interpret this data to predict flooding. Machine learning(ML) provides new ways to utilize this data efficiently. Varying approaches including artificial neural networks, decision trees including random forests, long short-term memory (LSTMs) and support vector machines (SVMs) have been useful in flood prediction [1][2], and the accuracy of these is improved by optimizing the models and combining the models through algorithm ensemble [1].')
st.write('Flood risk is often evaluated based on rainfall and water proximity. One of the datasets we will be using is the Georgia rainfall indicator that catalogs rainfall totals over monthly periods. We expect the anomaly feature will be utilized here (https://data.humdata.org/dataset/geo-rainfall-subnational). We will also utilize FEMA Flood insurance information to help determine the location and extent of flooding. The latitude, longitude, and date of loss features will be used (https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2). Note that these may change based on the status of the NCEI Archive which houses varying rainfall and flood disaster data sets and is currently inaccessible due to damage from hurricane Helene.')

st.header('Problem Definition')
st.write('To address the need for flooding predictors, we will first compile different types of weather and climate data. Then, we will reduce the compiled data\'s dimensionality through principal component analysis (PCA) and finally, normalize the data to improve readability and clarity. We plan to analyze the data using LSTMs, SVMs, and random forest models. These each present their own unique advantages.')

st.header('Methods')
st.write('We plan to train our model using data from various sources: climate, weather, location, and infrastructure datasets. Weather datasets often have missing or incomplete data. First, we will handle this missing data via imputation or deleting datapoints [3]. Since these features have strong relationships with each other, combining them or reducing dimensionality will improve its performance [4]. Lastly, normalizing this data will ensure no single feature dominates the model due to scale differences [5].')
st.write('Time is a crucial factor in flood prediction, so we can use sequential models like LSTMs to capture long-term dependencies in time-series data [6]. Because of our data\'s complex and non-linear relationships, SVMs would be perfect to handle this data [7]. In addition, SVMs are adept at handling binary classification tasks, which this problem is. Also given the sensitivity of the problem, the Random Forest model can offer high accuracy and handle noise and outliers making this a reliable option [7]. All of these models are supervised learning models that are the best options to address this problem.')

st.header('Results and Discussion')
st.write('We will measure this based on accuracy, F1 score, and AUC [8]. Accuracy would provide a measure of how well a model correctly predicts flood and no-flood events, ensuring a reliable assessment of overall model performance. We aim to achieve a >0.9 accuracy rate. The F1 score calculates the weighted harmonic mean of precision and recall, which is nice to evaluate the performance of a classification model. This is a good metric to ensure the model is accurately detecting flood events while minimizing false alarms. We aim to achieve >0.8 F1 score. AUC-ROC (Area Under a Curve) can assess the model\'s ability to distinguish between flood and no-flood events across various thresholds, which makes it a good metric to make our model reliable in real-world scenarios. We aim to achieve >0.8 AUC score.')
st.write('The model is expected to achieve a high accuracy and F1 score due to the history of floods within Georgia, which should make it easier for the model to distinguish between flood and no-flood events.')

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
st.write('https://gtvault-my.sharepoint.com/:x:/g/personal/rotsuka6_gatech_edu/EUrn1ZUCChlOv21P4fYxDuoB7fneo5ltPQRLfUX1f70xLw?e=oIgIYj')
st.header('Contribution Table')
st.write('https://docs.google.com/spreadsheets/d/1VFdCU-zb4ytwHfnp1iXquoY1NM_RtAKTSaC6XjDmqpY/edit?usp=sharing')