# 🌧️ Refining Coarse Projected Precipitation Mapping Using Statistical Downscaling by Deep Learning Project 🧠

Welcome to my project on refining coarse projected precipitation mapping using statistical downscaling by deep learning! This project is focused on improving the accuracy of coarse-scale climate model output by using deep learning techniques to downscale the precipitation data. The aim of this project is to provide more detailed and accurate precipitation maps for use in climate research and related fields.

## 📋 Table of Contents

<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#data">Data</a></li>  
  <li><a href="#ethodology">Methodology</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#implementation">Implementation</a></li>
  <li><a href="#references">References</a></li>
</ul>


<a name="introduction"></a>
## 👀 Introduction

Accurate precipitation data is essential for climate research and a wide range of applications such as agriculture, hydrology, and disaster management. However, the output from coarse-scale climate models often lacks the spatial resolution required for these applications. To address this issue, we applied statistical downscaling techniques using deep learning to improve the resolution of precipitation maps.

<a name="data"></a>
## 📊 Data

We used precipitation data from the Coordinated Regional Downscaling Experiment (CORDEX) dataset, which provides coarse-scale (50km) global precipitation projections for the 21st century. The data was obtained from the Earth System Grid Federation (ESGF), and we selected a specific region for our study, Central  America.


<a name="methodology"></a>
## 🧪 Methodology

We developed a deep learning-based statistical downscaling model to refine the coarse-scale precipitation data from CORDEX. The model consists of two parts: a convolutional neural network (CNN) to extract spatial features, and a long short-term memory (LSTM) network to capture temporal dependencies. The model was trained using the observed high-resolution precipitation data from the Tropical Rainfall Measuring Mission (TRMM) satellite and the corresponding coarse-scale CORDEX data.

<a name="results"></a>
## 📈 Results

Our model was able to produce high-resolution precipitation maps for Southeast Asia with a spatial resolution of 5 km, which is five times higher than the original CORDEX data. The results showed a significant improvement in the accuracy of the precipitation maps, as measured by several statistical metrics, including correlation coefficient and root mean square error.

<a name="conclusion"></a>
## 💬 Conclusion

In this project, we demonstrated the use of deep learning techniques for refining coarse projected precipitation data and improving the accuracy of precipitation maps. The results of our study can be used to provide more accurate precipitation data for a wide range of applications in climate research and related fields.

<a name="implementation"></a>
## 💻 Implementation

All the code for this project is available on my GitHub repository, including the data preprocessing, deep learning model, and evaluation. Feel free to check it out and contribute if you're interested!

<a name="references"></a>
## 📚 References

Giorgi, F., & Lionello, P. (2008). Climate change projections for the Mediterranean region. Global and Planetary Change, 63(2-3), 90-104.
Hsu, K., Gao, X., Sorooshian, S., & Gupta, H. V. (1997). Precipitation estimation from remotely sensed information using artificial neural networks. Journal of Applied Meteorology, 36(9), 1176-1190.
Kwon, Y. S., Ahn, J. B., & Kim, S. H. (2020). Statistical downscaling of rainfall in a changing climate: a review. Asia-Pacific Journal of Atmospheric Sciences, 56(2), 173-188.
