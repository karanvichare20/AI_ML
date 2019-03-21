#!/usr/bin/env python
# coding: utf-8

# $$\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\Yv}{\mathbf{Y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\betav}{\mathbf{\beta}}
# \newcommand{\gv}{\mathbf{g}}
# \newcommand{\Hv}{\mathbf{H}}
# \newcommand{\dv}{\mathbf{d}}
# \newcommand{\Vv}{\mathbf{V}}
# \newcommand{\vv}{\mathbf{v}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\Sv}{\mathbf{S}}
# \newcommand{\Gv}{\mathbf{G}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\Zv}{\mathbf{Z}}
# \newcommand{\Norm}{\mathcal{N}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}
# \newcommand{\dimensionbar}[1]{\underset{#1}{\operatorname{|}}}
# \newcommand{\dimensionbar}[1]{\underset{#1}{\operatorname{|}}}
# \newcommand{\grad}{\mathbf{\nabla}}
# \newcommand{\ebx}[1]{e^{\betav_{#1}^T \xv_n}}
# \newcommand{\eby}[1]{e^{y_{n,#1}}}
# \newcommand{\Tiv}{\mathbf{Ti}}
# \newcommand{\Fv}{\mathbf{F}}
# \newcommand{\ones}[1]{\mathbf{1}_{#1}}
# $$

# <h1 align="center">Letter Recognition(English Alphabet) using KNN Algorithm</h1>
# <center><b>---------------------------------------------------------------------</b></center>
# <center><b>Submitted By: Karan Milind Vichare (11010399) </b></center>
# <center><b>Under the guidance of Dr. Simon Ziegler </b></center>
# <center><b>Applied Artificial Intelligence (Dec 2018)</b></center>
# <center><b>@ SRH Hochschule Heidelberg </b></center>

# <a id='Top'></a>
# ## Table of Contents
# 
# * <a href='#Section1'>1. Introduction</a>
# * <a href='#Section2'>2. Motivation for this Project</a>
# * <a href='#Section3'>3. The Methodology</a>
# * <a href='#Section4'>4. Understanding Data Set</a>
# 
# * <a href='#Section5'>5. Theory Behind KNN Algorithm</a>
#     
# * <a href='#Section6'>6. Utilities</a>
# 
# * <a href='#Section7'>7. Implementation</a>
#     * <a href='#Section71'>7.1 Import Packages</a>
#     * <a href='#Section72'>7.2 Data Analysis</a>
#     * <a href='#Section73'>7.3 Data Partition</a>
#     
# * <a href='#Section8'>8. Train and Evaluate Models</a>
#     * <a href='#Section81'>8.1 K-Nearest Neighbors Classification</a>
# * <a href='#Section9'>9. Summary</a>

# <a id='Section1'></a>
# ## I. Introduction

# This Project was completed in the module of Applied Artificial Intelligence in order to understand and implement K-NN classification Algorithm of Machine Learning. It classifies all the samples of black & white rectangular pixel as in the dataset that is used to display one of the twenty-six letters present in English alphabets. In order to implement the algorithm I have used different steps by primarily focusing on training a set of models using scikit-learn available libraries for applying K-NN Algorithm and the and performance was measured.

# <a id='Section2'></a>
# ## II. Motivation for this Project

# Looking at the tremendous progress in Image-Processing technology, there are many applications available which are focusing to achieve close and accurate character recognition. Hence, by applying intelligence to it to make it learn by its own, this feature is expected to grow functionally and non-functionally. However, this project only focuses on identifying or classifying English Alphabets given on a rectangular pixel displays as in the data set.

# <a id='Section3'></a>
# ## III. Approach

# As mentioned above, K-NN classification algorithms is considered for categorizing number of given rectangular pixels as one of the Letters from English Alphabet. Also, further sections of this notebook will give brief explanation about the implementation of the above mentioned classification algorithm. And finally, it will focus on the detail analysis of the results/observations achieved upon application of the given learning algorithm to the data set of "Letter Recognition" which is obtained from UCI Machine Learning Library.
# 
# #### Required Files: Following files are required to be present in the current working directory.
# * *letter-recognition.data*
# 

# <a id='Section4'></a>
# ## IV. Understanding Data Set:
# 
# The data set which is used for this Course Work is downloaded from [Letter Image Recognition Data](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition) available in UCI Machine Learning Library. It is a huge data set consisting of around 20,000 samples of black & white rectangular pixels with 16 different/unique attributes (numericals) and one target/output variable. As explained on their website, the value which is an integer that is generated from scanning raster of these letters that highlights useful characteristics and statistical features associated to those pixels. With the above information our model will be trained so that it categorizes the letters accurately. 
# 
# Features of the dataset:
# 1.	lettr	capital letter	(26 values from A to Z) 
# 2.	x-box	horizontal position of box	(integer) 
# 3.	y-box	vertical position of box	(integer) 
# 4.	width	width of box	(integer) 
# 5.	high height of box	(integer) 
# 6.	onpix	total # on pixels	(integer) 
# 7.	x-bar	mean x of on pixels in box	(integer) 
# 8.	y-bar	mean y of on pixels in box	(integer) 
# 9.	x2bar	mean x variance	(integer) 
# 10.	y2bar	mean y variance	(integer) 
# 11.	xybar	mean x y correlation	(integer) 
# 12.	x2ybr	mean of x * x * y	(integer) 
# 13.	xy2br	mean of x * y * y	(integer) 
# 14.	x-ege	mean edge count left to right	(integer) 
# 15.	xegvy	correlation of x-ege with y	(integer) 
# 16.	y-ege	mean edge count bottom to top	(integer) 
# 17.	yegvx	correlation of y-ege with x	(integer)
# 
# As from above feature details, we have 17 different attributes available in the data set that describes unique statistical behaviour of the pixels. From these 17 features, first column is our target variable that has 26 unique values i.e A to Z. 
# 
# |Class|No.of Samples|Class|No.of Samples|Class|No. of Samples|Class|No. of Samples|
# |---|---------------|----|-----------|---|--------|---|--------|
# |A|789|I|755|Q|783|Y|786|
# |B|766|J|747|R|758|Z|734|
# |C|736|K|739|S|748|---|---|
# |D|805|L|761|T|796|---|---|
# |E|768|M|792|U|813|---|---|
# |F|775|N|783|V|764|---|---|
# |G|773|O|753|W|752|---|---|
# |H|734|P|803|X|787|---|---|
# 
# <h4 align="center">Table 1: (20,000 Sample Distribution)</h4>

# <a id='Section5'></a>
# ## V. Theory Behind KNN Algorithm
# <br>Before we apply K-NN ALgorithm on our data set, let us understand its implementation with some theory.
# 
# 
# ### K-Nearest Neighbors Classification:
# 
# K-NN is one of the simplest Machine-Learning Algorithms which helps to identify the class of a given sample based on the data samples that are nearest neighbors to it. As for the value specified for K, the trained model will look for its 'K' nearest neighbors which is done by computing the distances and then choosing the closest ones independent of which classes those samples belong to. 
# For instance, when K = 3, the algorithm looks for 3-Nearest Neighbors of any data points, and the maximum voted class from the neighbouring samples will be the class of that samples that will be predicted.
# 
# <img src = "https://upload.wikimedia.org/wikipedia/commons/e/e7/KnnClassification.svg">
# 
# As explained in the above example, in which we have to predict the class of the green point. Now as explained in K-NN, if we consider K = 3, the algorithm will look for 3 of the closest points as per the distance from the green data point. Here, we have two points that belong to red class 1 point that belongs to blue color class blue. Therefore, we can say that the class of green data points is going to be of the red class. That is the Prediction!!!!
# 
# <img src = "https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/09/k-Nearest-Neighbors-algorithm.png">
# 
# 
# From this image, we can say that the background color represents the class of the points in that region. This means that any points lying on blue background will belong to class Blue and points lying on green background will belong to Green Class.
# 

# Click <a href='#Top'>Top</a> to go to top of the page.

# <a id=Section6></a>
# ## VI. Required Utilities
# 
# This section shows some functions which are required further in the implementation sections of this project.

# **1) LoadCharacterData(filename):**
# - **Description:** This function reads the file specified in the parameter and modifies the string values in the class column with corresponding class numbers. It starts by replacing the letters A to Z by numbers 1 to 26 as shown below.
# 
# |Class|Class Number|Class|Class Number|Class|Class Number|
# |---|---------------|----|-----------|---|--------|
# |A|1|I|9|Q|17|
# |B|2|J|10|R|18|
# |C|3|K|11|S|19|
# |D|4|L|12|T|20|
# |E|5|M|13|U|21|
# |F|6|N|14|V|22|
# |G|7|O|15|W|23|
# |H|8|P|16|X|24|
# |---|---|---|---|Y|25|
# |---|---|---|---|Z|26|
# 
# - **Parameters:**<br>
# filename: Name of the data file to be read for analysis. The file specified in the filename parameter should be present in the current directory.
# - **Return value:** This function returns following values:<br>
# modifiedDataFrame:      Data Frame consisting of modified class values (A -> 1 .... Z -> 26). <br>
# dataFrame: Original Data Frame

# In[1]:


def LoadCharacterData(fileName):
    
    dataFrame = pd.read_csv(fileName, delimiter=',', header = None, 
                            names = ["Letter","xBox", "yBox", "Width", "Height", "OnPix", "xBar", 
                                                                             "yBar", "x2Bar", "y2Bar", "xyBar", "x2yBar", "xy2Bar", "xEdge",
                                                                             "xEdgeCORy", "yEdge", "yEdgeCORx"])            # Data Frame containing original values


    modifiedDataFrame = pd.read_csv(fileName, delimiter=',', header = None, 
                                    names = ["Letter","xBox", "yBox", "Width", "Height", "OnPix", "xBar", 
                                                                             "yBar", "x2Bar", "y2Bar", "xyBar", "x2yBar", "xy2Bar", "xEdge",
                                                                             "xEdgeCORy", "yEdge", "yEdgeCORx"])    # Data Frame containing modified targets
    
    classes = np.unique(dataFrame["Letter"])                                          # Unique Classes present in DataSet
    print ('{:s} {:}'.format('Unique Classes in the given data set:',classes))
    
    modifiedDataFrame["Letter"].replace({classes[0] : 1, classes[1]: 2, classes[2]: 3,
                                  classes[3] : 4, classes[4]: 5, classes[5]: 6,
                                  classes[6] : 7, classes[7]: 8, classes[8]: 9,
                                  classes[9] : 10, classes[10]: 11, classes[11]: 12,
                                  classes[12] :13, classes[13]: 14, classes[14]: 15,
                                  classes[15] :16, classes[16]: 17, classes[17]: 18,
                                  classes[18] :19, classes[19]: 20, classes[20]: 21,
                                  classes[21] :22, classes[22]: 23, classes[23]: 24,
                                  classes[24] :25, classes[25]: 26},inplace=True)
   
    return modifiedDataFrame,dataFrame,classes                        # return modifiedDataFrame and OriginalDataFrame along with classes
    


# **2) percentCorrect(predictedTargets, actualTargets)**
# - **Description:** This function computes the percentage of correct predictions based on the specified predicted and actual target classes.<br>
# - **Parameters:**<br>
# predictedTargets: Prediction obtained from the output of Use Function<br>
# actualTargets: Target values array<br>
# - **Return value:** This function returns percentage of correct prediction for a given model

# In[2]:


def percentCorrect(predictedTargets, actualTargets):
    return np.sum(predictedTargets.ravel()==actualTargets.ravel()) / float(len(actualTargets)) * 100


# Click <a href='#Top'>here</a> to go to top of the page.

# <a id='Section7'> </a>
# ## VII. Implementation
# 
# This section will analyze and implement different machine learning classification algorithm to train the model to classify the pixel displays in to 26 different classes. Initial subsections will focus on modifying the data frame to the appropriate format by adding header row. Next, we will focus on analyzing the data set and plotting relationship graphs among various parameters. Later subsection will focus on the actual implementation of different classification algorithms and performing a bunch of experiments by varying different parameters associated with each of the algorithms.

# <a id='Section71'> </a>
# ### 1. Import Packages:
# 
# Let's first import all the necessary packages required for the successful implementation of the above mentioned algorithms.

# In[3]:


import numpy as np               # To deal with arrays
import pandas as pd              # Standardized reading of data set
import matplotlib.pyplot as plt  # Plot graphs and figures
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.neighbors import KNeighborsClassifier # K Nearest Neighbors Classification


# <a id='Section72'> </a>
# ### 2. Data Analysis:
# 
# #### 2.1) Read Data Set:
# Now, we will first read the data using pandas package. The read command from pandas would return a dataframe which can be further converted to numpy array using "values" attributes. The dataframe is obtained using ```LoadCharacterData()``` function defined in **above section** of this notebook.
# 
# The values returned by LoadCharacterData function will be stored in ```mDF, oDF``` & ```_class``` variables. <br>
# 
# * ```mDF```: Modified Data Frame with Integer Class Labels<br>
# * ```oDF```: Original Data Frame with text Class Labels<br>
# * ```_class```: Class Labels

# In[4]:


mDF,oDF,_class = LoadCharacterData('letter-recognition.data')   # Returns Modified & Original DF.


# In[5]:


oDF.head()


# In[6]:


mDF.head()


# In[7]:


_class


# #### 2.2) Data Distribution Plot:
# As the first column of the data frame returned above contains the categories for each of the samples, we will plot it to see how the classes are distriuted throughout the samples.

# In[8]:


plt.figure(figsize=(10,8))
plt.plot(mDF.values[:,0], 'bo', markersize=4)
plt.ylabel("Class Name A-1 to Z-26")
 
plt.xlabel("Sample Index")


# ##### Observations: 
# *As seen in the above plot, the classes are well distributed accross 20000 samples. There are in total 26 classes each representing a letter from the English Alphabet. From the above plot, we dont get a clear picture of how many samples are present in each of the categories. Hence, there is a need to perform further analysis.*

# #### 2.3) Data Set Verification:
# 
# This section will verify if the modified dataframe ```mDF```, which consists of class labels as integers instead of letters, is correctly modified and has same number of samples for each class as sample count in the original class.

# In[9]:


dataOrig = oDF.values       # Convert original dataframe to Numpy Array
data = mDF.values           # Convert modified dataframe to Numpy Array

text = _class              
for integer in np.arange(0,26,step = 2):
    value = integer + 1
    print('{:s} {:s} {:} {:s} {:s} {:}'.format("No. of Samples in Class ",text[integer],len(np.where(dataOrig[:,0] == text[integer])[0].tolist()),
                                               "| No. of Samples in Class ",text[integer+1],len(np.where(dataOrig[:,0] == text[integer+1])[0].tolist())
                                               ))
    print('{:s} {:d} {:} {:s} {:d} {:}'.format("No. of Samples in Class ",value,len(np.where(data[:,0] == value)[0].tolist()), 
                                               "| No. of Samples in Class ",value+1,len(np.where(data[:,0] == value+1)[0].tolist())
                                              ))
    print(" ")


# ##### Observations:
# *From the above results, we can see that we have successfully been able to modify the string labels of all the classes to integer labels. As the number of samples in class with text labels is matching with that of the class with integer labels, we can coonclude that the data is modifed correctly*

# #### 2.4) Extract Targets
# As seen in Section-IV, there are 16 features and one targets. In this section, we will seperate out the targets from the feature vectors and plot them against each other to observe any general trend.

# In[10]:


X = data[:,1:]                 # Extract Input Features
T = data[:,0].reshape(-1,1)   # Extract Classes

print('{:s} {:}'.format("Shape of Input Matrix (X)  :-",X.shape))
print('{:s} {:}'.format("Shape of Target Matrix (T) :-",T.shape))


# #### 2.6) Input Features Vs. Target
# 
# Now, we will be plotting each of the columns of input features with respect to the target column in order to understand how the samples are distributed within a particular feature.

# In[11]:


columnTitle = mDF.columns.tolist()
plt.figure(figsize=(25,35))

for i in range(0, 16):
    plt.subplot(6,3, i+1)
    plt.plot(X[:,i],T[:],'go')
    plt.ylabel('Class')
    plt.title('{:s}{:d}{:s}{:s}{:s}'.format("Figure", i+1,": ", columnTitle[i+1], " Vs. Target"))
    plt.xlabel(columnTitle[i+1])

plt.tight_layout()


# ##### Observations:
# 
# *From the above plots, following observations can be made:*
# * *For figure 1, there are very few samples which have higer value of xBox. Most of the samples are present in the lower range of xBox i.e from  range 0 to 13. Similar conclusion can be made for plots in figures 3, 4, 9 & 13.*
# * *For figure 2, the samples seem to be uniformly distributed accros the range of 0 to 15 thus covering the entire gamut.*
# * *For figure 7, 10 & 14, large number of samples are mostly clustered in the range of 3 to 13*
# * *The parameters plotted in 7, 10 & 14 are basically associated with mean and correlation statistics.
# 
# Click <a href='#Top'>here</a> to go to top of the page.

# <a id='Section73'></a>
# ### 3) Data Partitions:
# 
# This section focuses on performing random partitioning of data. Here, we will be using partition function defined in mlutilities. The definition for this function is also shown for reference in *Section VI-1* of this notebook.
# 
# As mentioned earlier, we will be performing stratified partioning on data in order to get equal proportion of samples from each of the twenty-six categories in training and testing. The data would be divided in such a way that we get 80 % samples for training and remaining 20 % samples for testing the algorithms. As we are working on a classification problem, it is important to have equal proportion of the samples from each of the class so as to train the models for all the classes without any bias.

# In[12]:


T.shape, X.shape # Of these 20000 samples, 80% would be used for training and remaining 20 percent for testing.


# In[26]:


#Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, Ttrain, Ttest = train_test_split(X, T, test_size = 0.2, random_state = 0)

print('{:s} {:}'.format("Shape of Xtrain :-",Xtrain.shape))
print('{:s} {:}'.format("Shape of Ttrain :-",Ttrain.shape))
print('{:s} {:}'.format("Shape of Xtest :-",Xtest.shape))
print('{:s} {:}'.format("Shape of Ttest :-",Ttest.shape))


# ##### Before applying partitioning:
# 
# |Parameter|Data Summary|
# |-----|--------------|
# |No. of rows in X   |20,000|
# |No. of columns in X   |16|
# |No. of rows in T   |20,000|
# |No. of columns in T   |1|
# 
# ##### Training Data Summary:
# 
# |Parameter|Data Summary|
# |-----|--------------|
# |No. of rows in Xtrain   |16000|
# |No. of columns in Xtrain   |16|
# |No. of rows in Ttrain   |16000|
# |No. of columns in Ttrain   |1|
# 
# ##### Testing Data Summary:
# 
# |Parameter|Data Summary|
# |-----|--------------|
# |No. of rows in Xtest   |4000|
# |No. of columns in Xtest   |16|
# |No. of rows in Ttest   |4000|
# |No. of columns in Ttest   |1|
# 

# #### Verify data after Partitioning:

# In[14]:


classInT = np.unique(T)
print('   Class  #Samples')
for i in classInT:
    print('{:5g} {:10d}'.format(i, np.sum(T==i)))

print('\nPartitions in Training Set')

classInTrain = np.unique(Ttrain)
print('   Class  #Samples')
for i in classInTrain:
    print('{:5g} {:10d}'.format(i, np.sum(Ttrain==i)))

print('\nPartitions in Test Set')
    
classInTest = np.unique(Ttest)
print('   Class  #Samples')
for i in classInTest:
    print('{:5g} {:10d}'.format(i, np.sum(Ttest==i)))


# ##### Observations:
# *From the above output of partition function, we can see that each of the training and testing sets have equal proportion of samples from each of the classes. In other words, the ratio of number of samples of different classes is approximately the same for training and testing partitions. For example, the ratios ```789/766, 766/736 & 736/805```  are approximately equal to ```631/613, 613/589, 589/644``` for training set and ```158/153, 153/147 & 147/161``` for testing set. Hence, we can say that samples from each of the classes are proportinally distributed in training and testing data sets. This will ensure that the proportion of samples from each class to train the model is approximately equal to the proportion of samples used to test the model and thus giving us unbiased results.*
# 
# Click <a href='#Top'>here</a> to go to top of the page.

# <a id='Section8'> </a>
# ## VIII. Train & Evaluate Models:
# 
# In this section, we will be training different classification learning models and evaluate their performances in differnet aspects such as accuracy, execution time and resources required.
# 
# <a id='Section81'> </a>
# ### 8.1) K-Nearest Neighbors Classification:
# 
# As seen in the theory, KNN is a simple machine learning model that will be able to accurately classify the data if the features are very distinct. In this section, we will use the KNN model with Letter Recognition data set read above.

# ### 1) Training 1 (K = 1)
# 
# #### 1.1) Instantiate and Train KNN Model with K = 1:
# 
# KNeighborsClassifier module from scikit learn will be used and the parameter to the constructor of this class will be n_neighbors with value '1'.

# In[15]:


knn = KNeighborsClassifier(n_neighbors=1)   # KNN object Instantiation
print (knn)


# The above line creates an instance of KNeighborsclassifier class. This object will perform K Nearest Classification looking for one nearest neighbors. The nearest samples will be identified based on minkowski distance that is a default value for metric parameter.

# Now that we have the knn classifier object, let's train the model on the training input and target matrices.

# In[16]:


knn.fit(Xtrain, Ttrain.ravel())


# #### E.1.2) Use KNN-1 Model:
# 
# Once the model is trained on training samples, we need to use it to obtain the predictions on training and testing samples. The predict method from scikit learn classifiers will return a numpy array of predicted response values which is then stored in Ytrain_KNN for training predictions and Ytest_KNN for predictions on new samples.

# In[17]:


Ytrain_KNN = knn.predict(Xtrain)
Ytest_KNN = knn.predict(Xtest)


# #### E.1.3) Plot Predictions on Training and Testing Samples:

# By plotting the predictions obtained on training and testing samples, we will get some idea of how well the model is performing.

# In[18]:


Ytrain_KNN = Ytrain_KNN.reshape(-1,1)
Ytest_KNN = Ytest_KNN.reshape(-1,1)

plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(np.hstack((Ttrain, Ytrain_KNN)), 'o-', alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Predictions on Training Data')
plt.legend(('Actual Targets', 'Predicted Targets'), loc='lower right')

plt.subplot(1, 2 ,2)
plt.plot(np.hstack((Ttest, Ytest_KNN)), 'o-', alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Predictions on Test Data')
plt.legend(('Actual Targets', 'Predicted Targets'), loc='right')


# #### E.1.5) Percentage Accuracy for KNN (K = 1):

# In[19]:


print('{:s} {:}'.format('Percentage of Correct Prediction with KNN-1 on Training Data :',percentCorrect(Ytrain_KNN, Ttrain)))
print('{:s} {:}'.format('Percentage of Correct Prediction with KNN-1 on Test Data     :',percentCorrect(Ytest_KNN, Ttest)))


# ##### Observations:
# * *Looking at the predictions plot in E.1.3, one can say that the KNN model with K = 1 performs very well. *
# * *We get a 100 % accuracy on using this model on training samples. This is because, in order to make a predictions, the model looks for k sample nearest to it and based on the majority in the response values, the prediction is made. However, in K = 1 to make a prediction for any sample in the training set, this model will search for one nearest neighbor and it will find the exact same sample as its nearest one, every time. As the model is trained on training set, and we use the same data to test the model, this model will make [Predictions with 100 percent accuracy.*
# * *From the above point, it becomes obvious to not to use the same data to train as well as test the model. Hence, we perform testing on the test data in Xtest.*
# * *The KNN model with K = 1 seems to perform well on new data as the percentage of predction accuracy we have got is 96.15%
# 
# Click <a href='#Top'>here</a> to go to top of the page.

# ### E.2) Training 2 (KNN = 5)
# 
# Let's try using KNN model with 5 nearest neighbors.

# #### E.2.1) Instantiate & Train KNN model with K = 5:

# In[20]:


knn = KNeighborsClassifier(n_neighbors=5)   # KNN object Instantiation
knn.fit(Xtrain,Ttrain.ravel())


# #### E.2.2) Use KNN-5 Model

# In[21]:


Ytrain_KNN5 = knn.predict(Xtrain)
Ytest_KNN5 = knn.predict(Xtest)


# #### E.2.3) Plot Predictions on Training and Test Data:

# In[22]:


Ytrain_KNN5 = Ytrain_KNN5.reshape(-1,1)
Ytest_KNN5 = Ytest_KNN5.reshape(-1,1)

plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(np.hstack((Ttrain, Ytrain_KNN5)), 'o-', alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Predictions on Training Data')
plt.legend(('Actual Targets', 'Predicted Targets'), loc='lower right')

plt.subplot(1, 2 ,2)
plt.plot(np.hstack((Ttest, Ytest_KNN5)), 'o-', alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Predictions on Test Data')
plt.legend(('Actual Targets', 'Predicted Targets'), loc='right')


# #### E.2.4) Overall Percentage Accuracy:

# In[23]:


print('{:s} {:}'.format('Percentage of Correct Prediction with KNN-5 on Training Data :',percentCorrect(Ytrain_KNN5, Ttrain)))
print('{:s} {:}'.format('Percentage of Correct Prediction with KNN-5 on Test Data     :',percentCorrect(Ytest_KNN5, Ttest)))


# ##### Observations:
# * *From the above plots and results, we see that the percentage accuracy on trainign and testing samples is decreased slightly with KNN-5 model*
# * *As there is not much difference in the prediction accuracy of KNN-5 model, it can still be a good fit as the decision boundary with KNN-5 will be more generalized.*

# ### E.3) Training 3 (Multiple Values of K):
# 
# let's try to analyse the percentage accuracy for multiple values of K.

# #### E.3.1) Train & Evalue KNN with Multiple values of K:

# In[24]:


testAccuracy = []
trainAccuracy = []

for k in range (1,21):
    print('{:s} {:d}'.format("KNN with K = ", k))
    knn = KNeighborsClassifier(n_neighbors=k)   # KNN object Instantiation
    knn.fit(Xtrain,Ttrain.ravel())
    
    Ytrain_KNNK = knn.predict(Xtrain)
    Ytest_KNNK = knn.predict(Xtest)
    print('{:s} {:}'.format('Percentage of Correct Prediction on Training Data :',percentCorrect(Ytrain_KNNK, Ttrain)))
    print('{:s} {:}'.format('Percentage of Correct Prediction on Test Data     :',percentCorrect(Ytest_KNNK, Ttest)))
    print(" ")
    testAccuracy.append(percentCorrect(Ytest_KNNK, Ttest))
    trainAccuracy.append(percentCorrect(Ytrain_KNNK, Ttrain))


# #### E.3.2) Plot Prediction accuracy on training and testing data:

# In[25]:


plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(range(1,21), testAccuracy)
plt.xlabel('K-Value')
plt.ylabel('Percentage of Correct Predictions')
plt.title('Test Accuracy Vs K values')

plt.subplot(1,2,2)
plt.plot(range(1,21), trainAccuracy)
plt.xlabel('K-Value')
plt.ylabel('Percentage of Correct Predictions')
plt.title('Train Accuracy Vs K values')


# ##### Observations:
# * *Form the above results and plots we see that the best value of k with maxium percentage accuracy on training and testing sample is 1. The next best value of K for which the prediction accuracy on test data is maximum is K = 5.*
# * *From the above graph, it is also clear that with increase in the number of nearest neighbors the model takes in to consideration, the vale of accuracy on training and test data reduces for this particular data set.*
# 

# Click <a href='#Top'>here</a> to go to top of the page.

# <a id='Section9'></a>
# ## IX. Summary:
# 

# The below table summarizes the important results obtained from the implementation of K-NN algorithm for different values of K.
# 
# |Algorithm|% Accuracy on Training Data|% Accuracy on Test Data|
# |---------|-----|-----|
# |KNN (K=1)|100.0|96.15|
# |KNN (K=5)|97.57|95.97|
# |KNN (K = 20)|94.42|94.27|

# * *The above table shows various values of percentage accuracies on training and test set for different classification algorithms.*
# * *The maximum percentage of correct predictions on training set is obtained with KNN = 1 
# * *However, as our goal is to achieve maximum percentage of correct predictions on new data, we must use the algorithm with maximum percentage accuracy on test set. This is achieved with KNN (K = 1) & the next algorithm with maximum percentage accuracy on test data is KNN with K = 5.*
# * *Another interesting point to observe is that as the value of K increases for a KNN, the percentage accuracy on the both training and test samples reduces.*
# * *In sum, K nearest neighbors algorithm with smaller values of K, typically 5, performs slightly better.
# 
# With this I was successfully able to achieve the objective of understanding and implementing K-NN Algorithm to classify the Letter Recognition Data set. 
