#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: b1fxb03

3) Time series with pi
-----------------------
Attached is a function genPiAppxDigits(numdigits,appxAcc) which returns an approximate value of pi
to numdigits digits of accuracy. appxAcc is an integer that controls the approximation accuracy, with
a larger number for appxAcc leading to a better approximation.
i) Fix numdigits and appxAcc to be sufficiently large, say 1000 and 100000 respectively.
Treat each of the 1000 resulting digits of pi as the value of a time series. Thus x[n]=nth digit
of pi for n=1,1000. Build a simple time series forecasting model (any model of your choice)
that predicts the next 50 digits of pi. Report your accuracy. Using your results, can
you conclude that pi is irrational? If so, how?

(bonus) Now let's vary appxAcc to be 1000,5000,10000,50000,100000 with fixed numdigits=1000. You thus
have 5 time series, each corresponding to a value of appxAcc. Can you find the pairwise correlation
between each of the time series?

#
def genPiAppxDigits(numdigits,appxAcc):
	import numpy as np
	from decimal import getcontext, Decimal
	getcontext().prec = numdigits
	mypi = (Decimal(4) * sum(-Decimal(k%4 - 2) / k for k in range(1, 2*appxAcc+1, 2)))
	return mypi
"""
import pandas as pd
import numpy as np
from datetime import date
from os import chdir
import sys
import numpy as np
from decimal import getcontext, Decimal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

#%%
#setting directory
if sys.platform == "linux":
    directory = '/san/RDS/Work/nielsen/Projects/Fatima/Atlanta_Fed/quant_spec_coding'
    
else:
    directory = "//rb/B1/NYRESAN/RDS/Work/nielsen/Projects/Fatima/Atlanta_Fed/quant_spec_coding"
    
chdir( directory )

 #%%  

 #setting the seed
 np.random.seed(42)
 
 #defining the function
def genPiAppxDigits(numdigits,appxAcc):
    getcontext().prec = numdigits
    mypi = (Decimal(4) * sum(-Decimal(k%4 - 2) / k for k in range(1, 2*appxAcc+1, 2)))
    return mypi

#%% predicting the next 50 digits 

# Example sequence of digits for numdigits = 1000 and appxAcc = 100000
pi_value = genPiAppxDigits(1000,100000)
sequence = str(pi_value)[0:]

# Create a mapping from digits to integers
unique_digits = sorted(set(sequence))
digit_to_int = {digit: i for i, digit in enumerate(unique_digits)}
int_to_digit = {i: digit for digit, i in digit_to_int.items()}
num_unique_digits = len(unique_digits)

# Prepare input data and labels
sequence_length = 10  # Number of digits in each input sequence
X = []
y = []
for i in range(len(sequence) - sequence_length):
    input_seq = sequence[i:i+sequence_length]
    output_digit = sequence[i+sequence_length]
    X.append([digit_to_int[digit] for digit in input_seq])
    y.append(digit_to_int[output_digit])

X = np.array(X)
y = np.array(y)

# Building the RNN model
model = Sequential([
    SimpleRNN(64, input_shape=(sequence_length, 1), activation='relu'),
    Dense(num_unique_digits, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X.reshape(-1, sequence_length, 1), y, epochs=1000)

# Generate predictions for the next 50 digits
predicted_sequence = sequence # Initial sequence for prediction
for _ in range(50):
    input_seq = predicted_sequence[-sequence_length:]
    input_seq_int = np.array([[digit_to_int[digit]] for digit in input_seq])
    predicted_digit_int = model.predict(input_seq_int.reshape(1, sequence_length, 1)).argmax()
    predicted_sequence += int_to_digit[predicted_digit_int]

print("Predicted sequence:", predicted_sequence)

#%% Reporting the accuracy 
#looking to compare only the last 50 digits
calculated_pi = genPiAppxDigits(1050,100000)
calculated_sequence = str(calculated_pi)[0:]
truncated_calculated_sequence = str(calculated_pi)[1000:1050]

truncated_predicted_sequence = predicted_sequence[1000:1050]

#this calculates the accuracy of only the last 50 digits
accuracy_last50 = accuracy_score(list(truncated_calculated_sequence), list(truncated_predicted_sequence))
print("Accuracy: ", accuracy_last50)

#this calculates the accuracy of all 1050 digits
accuracy_all = accuracy_score(list(calculated_sequence), list(predicted_sequence))
print("Accuracy: ", accuracy_all)
#More of a sanity check to confirm the first 1000 digits of the sequence are the same 
#and has a very high accuracy 

#%%
"""
Based on the results, the forecasting model only predicted a low percentage of 
the next 50 digits accurately. There is no guarantee that this will continue 
to be more accurate, even if we predict the next million numbers from now.
Since the accuracy of the model is low, there could be another model that is
more predictive, but it will still not be able to predict all the values of pi.
In this case, we cannot use the results to conclude pi is irrational because 
the model is limited and relies on patterns.
Since pi is irrational, the decimal representation goes without repeating. As a
result, the next digits cannot be predicted accurately using a finite model, but
it aids in our understanding that irrational numbers are not predictable and are
non-repeating in nature.
"""
#%% this is the bonus question that looks at 5 time series

# Generate the sequence using the genPiAppxDigits function
numdigits = 1000
appxAcc_values = [1000, 5000, 10000, 50000, 100000]

for appxAcc in appxAcc_values:
    pi_value = genPiAppxDigits(numdigits, appxAcc)
    pi_sequence = str(pi_value)[0:]

    # Convert the sequence into a numerical representation
    unique_digits = sorted(set(pi_sequence))
    digit_to_int = {digit: i for i, digit in enumerate(unique_digits)}
    int_to_digit = {i: digit for digit, i in digit_to_int.items()}
    num_unique_digits = len(unique_digits)
    numeric_sequence = [digit_to_int[digit] for digit in pi_sequence]

    # Create input and output sequences for training
    sequence_length = 10
    X = []
    y = []
    for i in range(len(numeric_sequence) - sequence_length):
        input_seq = numeric_sequence[i:i+sequence_length]
        output_digit = numeric_sequence[i+sequence_length]
        X.append(input_seq)
        y.append(output_digit)

    X = np.array(X)
    y = np.array(y)

    # Build the LSTM model
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, 1), activation='relu'),
        Dense(num_unique_digits, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X.reshape(-1, sequence_length, 1), y, epochs=100)

# Generate predictions for the next 50 digits
    predicted_sequence = sequence # Initial sequence for prediction
    for _ in range(50):
        input_seq = predicted_sequence[-sequence_length:]
        input_seq_int = np.array([[digit_to_int[digit]] for digit in input_seq])
        predicted_digit_int = model.predict(input_seq_int.reshape(1, sequence_length, 1)).argmax()
        predicted_sequence += int_to_digit[predicted_digit_int]

    print("AppxAcc: ", appxAcc, "Predicted sequence: ", predicted_sequence)

"""
I ran out of time to complete the bonus, so it is missing the pair wise 
correlation portion.
"""
