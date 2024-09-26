# The City College of New York, City University of New York
# Written by Grace McGrath and Ishmam Fardin
# July 2023 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore") # supresses all warnings 

# Personalized prognosis of Alzheimer's Disease progpression using ANNs
# Load the training data, Input X
training_X = 'training_data/gene_expressions.csv'
X = pd.read_csv(training_X)

# filters out unncessary information like age, name, etc.
# only focuses on gene expressions
X_in = X.filter(regex='x_at', axis=1) 
# print(X_in.head())

# Load the rest of the training data. Output Y
training_Y = 'training_data/mmse_score_slopes.csv'
Y = pd.read_csv(training_Y)
# print(Y.head())

# fit_transform converts all the values which are the slope of cognition 
# decline into values that are between 0 and 1
# using MinMaxScaler which does (original value - minimum) divided by 
# (maximum - minimum) to get the new value
# this is done to help the algorithm perform better and also leads to outliers 
# having less impact
# in reshape(-1,1) the -1 signifies the number of rows in the array and the 1 
# represents one column  reshapes converts the array into a 2D array
mmse_slopes = np.array(Y["m-coef(mmse-points/week)"]).reshape(-1,1)
Y_scaled = MinMaxScaler().fit_transform(mmse_slopes)

# hold all percent errors for each sample that was left out for LOOCV
percent_slope_errors = [] 
# hold all mmse score errors
percent_mmse_score_errors = []
# hold all predictions 
predictions = [] 
# hold predictd_mmse_scores
predicted_mmse_scores = []
# hold all actual slopes 
actual_slopes = [] 
# hold actual_mmse_scores
actual_mmse_scores = []
# hold all patient IDs 
patient_IDs = []

# Find the percent error for each sample that's left out 
# Utilize LOOCV to evaluate performance of model
for i in range(4):
  print("\n********************************")
  print(f"LEAVING OUT ROW {i} \n")

  # initialize a copy of all inputs to a new variable so that original 
  # variable does not get updated
  inputs = X_in.copy()
  
  # initialize a copy of all outputs that were scaled to a new variable so 
  # that original variable does not get updated
  outputs = Y_scaled.copy() 
  
  # save the input for test sample that will be used to see how effecient 
  # the model is
  LOOCV_input = inputs.loc[i] 
  
  # get the slope of test sample
  actual_slope = list(Y["m-coef(mmse-points/week)"])[i]
  
  # drop the row with test sample inputs
  inputs.drop(i,inplace=True) 
  
  # remove the output of test sample from the output array that will be used 
  #to train model
  outputs = np.delete(outputs, i)

  # create a model for the ANN
  model = keras.Sequential()
  print("Training Started")

  # remove / add hashtags to test architectures


  # Define and train model architecture 120-60-15-5-1
  
  # add a hidden layer that takes in 63 input features 
  # (the gene expressions that have x_at)
  # hidden layer has 120 neurons
  # dense connects all of it's neurons to the neurons in the layer after, 
  # and the layer before
  model.add(keras.layers.Dense(120,
                               kernel_regularizer='l1',
                               activation='relu',
                               input_shape=(63,), )
                               )

  # add another hidden layer with 60 neurons
  model.add(keras.layers.Dense(60,kernel_regularizer='l1', activation='relu' ))

  # add another hidden layer with 15 neurons 
  model.add(keras.layers.Dense(15, activation='relu'))

  # add another hidden layer with 5 neurons 
  model.add(keras.layers.Dense(5,activation='relu'))

  # add an output layer with one output (the slope of the cognitive decline)
  # activation function default to linear  
  model.add(keras.layers.Dense(1))
  
  # set optimizer to Adam which is a gradient descent method that updates 
  # the weights to minimize the loss
  # set loss to mean_absolute_error which does the average absolute 
  # difference between actual output and estimated output
  # set learning rate to 0.01, learning rate is the rate at which model 
  # paremeters are updated)  
  # metrics is utilized to see how the model is performing each epoch 
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                loss='mean_absolute_error', 
                metrics=['Precision']
                ) 

  # train the model for 500 epochs
  model.fit(inputs, outputs, epochs = 500, verbose=0)

  print("Training Finished\n")
   
  gene_expressions = LOOCV_input.to_frame() # converts series to dataframe
  
  # transpose rows/columns so that it is compatible with 
  # model input configuration
  gene_expressions_transposed = gene_expressions.transpose() 
  
  # make a prediction based on inputs
  prediction_scaled = model.predict(gene_expressions_transposed)
  
  # initialize the current patient ID to a variable
  patient_ID = X['patient-id'][i]
  patient_IDs.append(patient_ID)

  # find the minimum of the original output values
  mmse_score_min = Y['m-coef(mmse-points/week)'].min() 
  # find the maximum of the original output values
  mmse_score_max = Y['m-coef(mmse-points/week)'].max()
  # normalize the prediction 
  mmse_max_min_difference = mmse_score_max - mmse_score_min
  prediction = mmse_score_min + prediction_scaled*mmse_max_min_difference
  
  # calculate percent error
  rel_err = abs(((prediction - actual_slope )/actual_slope)*100)
  rel_error = round(float(rel_err[0][0]), 2)
  print("\nThe relative error is : {}%".format(rel_error))
  percent_slope_errors.append(rel_error) # adds percent error to list

  # Round the prediction and actual slope
  prediction = round(float(prediction[0][0]), 3)
  actual_slope = round(float(actual_slope), 3)
  print(f"The predicted MMSE Score slope is: {prediction} mmse-points/week")
  print("The actual slope is :", actual_slope)

  predictions.append(prediction) # adds prediction to list
  actual_slopes.append(actual_slope) # adds actual slope to list

  # Generate a plot of the predicted ouptut vs the actual
  
  # baseline is the score that the patient started with
  baseline = list(Y["m-baseline(mmse-points)"])[i] 
  
  # generate a sequence of evenly spaced numbers between 0 and 52 with 
  # a total of 7 values
  x = np.linspace(0, 52, 7) 
  
  # create a figure to hold all the plots, and makes it a white background
  # thats 8 by 6 inches
  plt.figure(figsize=(8, 6), facecolor='white') 
  
  plt.subplots_adjust(top=0.85)  # Adjust top margin for title

  # string representation of the equation for actual output with 
  # baseline initialized to a variable
  actual_equation = f"Actual: y = {actual_slope}x + {round(float(baseline),2)}"
  
  # string representation of the equation with actual output with 
  # baseline initialized to a variable
  rounded_baseline = round(float(baseline),2)
  prediction_equation = f"Predicted: y = {prediction}x + {rounded_baseline}" 
  
  # calculate the minimum y-value for the actual data
  actual_min = actual_slope*52+baseline
  actual_mmse_scores.append(round(float(actual_min),2))
  
  # calculate the minimum y-value for the predicted data
  predicted_min = prediction*52+baseline
  predicted_mmse_scores.append(round(float(predicted_min),2))
  
  # record percent_error_mmse_score
  percent_mmse_score_errors.append((predicted_min-actual_min)/actual_min)
  
  # Show equation of each line on the graph
  
  # add a text annotation to the plot at the coordinates (2, actual_min)
  # The text content is given by actual_equation, and it will be displayed 
  # in the color 'palevioletred'
  # The ha='left' and va='bottom' parameters specify the horizontal 
  # and vertical alignment of the text, respectively
  plt.text(2, actual_min+1, actual_equation, color='palevioletred', 
           ha='left', va='bottom')
  
  # add a text annotation at the coordinates (2, predicted_min)
  # The text content is given by prediction_equation, and it will be displayed 
  # in the color 'palevioletred'
  # The ha='left' and va='bottom' parameters represent horizontal and vertical 
  # alignment of the text
  plt.text(2, predicted_min, prediction_equation, color='skyblue', 
           ha='left', va='top')

  # plot the line of the actual output , sets color of line to palevioletred,
  # and for the legend, it adds a label title
  plt.plot(x, actual_slope*x + baseline, color='palevioletred', 
           label ='real MMSE score decline (best fit)')
  
  # plot the line of the predicted output , sets color of line to skyeblue,
  # and for the legend, it adds a label title
  plt.plot(x, prediction*x+baseline, color='skyblue', 
           label ='predicted MMSE score decline') 
  
  plt.xlabel('Weeks Since First Mini-Mental State Exam') # sets x axis label
  plt.ylabel('MMSE Score') # sets y axis label 
  plt.xlim([0, 52]) # sets x axis range from 0 to 52

  # set title 
  title = "MMSE Score Decline \n Predicted vs Actual \n Patient:"
  plt.title(title + patient_ID) 
  
  plt.legend() # shows the legend
  
  # save the graph into predictions directory
  plt.savefig(f'predictions/{patient_ID}.png') 

print("****************************")
print("LOOP DONE \n")

# List the percent error, slopes
for i in range(len(percent_slope_errors)):
    print("\nPatient ID:", patient_IDs[i])
    print(f'Slope of Cognitive Decline: Actual: {actual_slopes[i]}\
  Predicted: {predictions[i]}')
    print(f'Percent Slope Error: {percent_slope_errors[i]}')
    print(f'MMSE score at Week 52: Actual: {actual_mmse_scores[i]}\
  Predicted: {predicted_mmse_scores[i]}')
    print(f'Percent MMSE Score Error at Week 52:\
    {round(float(percent_mmse_score_errors[i]),2)}')

pct_sum = sum(percent_slope_errors)
len_pct_err = len(percent_slope_errors)
pct_sum_mmse = sum(percent_mmse_score_errors)
print(f'\nAverage % Slope Error: {round(float(pct_sum/len_pct_err),3)}%')
print(f'Average % MMSE Score Error at Week 52:\
 {round(float(pct_sum_mmse/len_pct_err),3)}%')