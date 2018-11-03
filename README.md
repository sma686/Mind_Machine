# Mind_Machine
Data Science and Machine Learning Projects
Project on Machine learning for classification problem when there are missing values.
Strategy that was executed in the code:
step1: Data cleaning and filling:In this step I have filled the missing numbers with linear interpollation and missing categorical variable cells with KNN method.
step2: standard scalar is used to normalize all the variables except the target
step3: Decision tree classification in used to this problem as it is giving more accuracy and performance. Logistic regression is also a high performance algorithm but low F1 score when compared to Decision Tree algorithm.
step4: predicted values for the test data  set are stored in the variable 'y_pred'
Extra code has been written in the comment section below. It can be used for measuring accuracy and F1 score of the prediction if test validation data is available.
As to not to loose the information of the data set feature scaling has not applied.

Execution steps:
Important Libraries to be installed: fancyimpute from conda install.
Set the working directory to the data containing folder and run the whole program for results.
