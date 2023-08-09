/*
2) Prediction + filtering
--------------------------

Attached are 3 files: xvalsSine.csv, cleanSine.csv and noisySine.csv. xvalsSine.csv contains
1000 x-values in the interval -pi/2 to pi/2. cleanSine.csv is a pure sine(x) function for the
x values mentioned earlier. noisySine.csv contains sine(x) corrupted by noise. 


i) Using xvalsSine.csv and cleanSine.csv as a labeled dataset (x,sine(x)) being (value,label) with a
random train/test split of 0.7/0.3, build an OLS regression model (you may want to use polynomial
basis of a sufficiently large order). 


(bonus) If you used the normal equations to solve the OLS problem, can you redo it with stochastic
gradient descent?
	easier to do this in python since stata doesnt have it built in
da

ii) Now, assume you are given the noisySine.csv as a time series with the values of 
xvalsSine.csv being the time variable. Filter the noisySine.csv data with any filter of your choice
and compare against cleanSine.csv to report the error.


(bonus) Can you code a Kalman filter to predict out 10 samples from the noisySine.csv data?
*/
*****************************************************************************
//Preamble
	clear all
	set seed 402

	di "`c(os)'"

	if "`c(os)'" == "Unix" {
	   global main /san/RDS/Work/nielsen
	}
	else if "`c(os)'" == "Windows" {
	   global main "\\rb.win.frb.org\B1\NYRESAN\RDS\Work\nielsen"
	}
	global project	${main}/Projects/Fatima/Atlanta_Fed/quant_spec_coding
	global code ${project}/code
	global output	${project}/output
	global data ${project}/data
	
*****************************************************************************
//Importing and merging the data for xvalsSine and cleanSine

	import delimited "/san/RDS/Work/nielsen/Projects/Fatima/Atlanta_Fed/quant_spec_coding/data/xvalsSine.csv"
	rename v1 x
	
	tempfile xvalues
	save `xvalues' 
	
	clear
	import delimited "/san/RDS/Work/nielsen/Projects/Fatima/Atlanta_Fed/quant_spec_coding/data/cleanSine.csv"
	rename v1 sine_x
	
	merge 1:1 _n using `xvalues' //merging by observation count 
	drop _merge
	
	tempfile x_and_clean_sine
	save `x_and_clean_sine'
	
*****************************************************************************
//splitting the sample into a random .7/.3 split 

	splitsample x sine_x, split(70 30) gen(sample)
   
*****************************************************************************	
//making polynomial x variables and finding the optimal polynomial for the regression
local power = 16
	forvalues power = 2/`power' {
		gen x_`power' = x^`power'
	}
   
*****************************************************************************	
//build an OLS regression model
	
	//regressing only the train data sample 
	reg sine_x x x_* if sample == 1
	local df = `e(df_r)'
	
	//making a variable to generate predictions on the test data 
	predict x_prediction if sample == 2
	
	sort x_prediction
	
	//16 is the best because when comparing sine_x and prediction, it has
	//the most matches after looping through powers of 2 through 20
	compare sine_x x_prediction
	
	//calculating accuracy of model using mean squared error 
	//https://stats.stackexchange.com/questions/41695/what-is-the-root-mse-in-stata
	gen difference_sine_x = sine_x - x_prediction
	gen squared_difference = (difference_sine_x)^2
	egen mean_square_error = total(squared_difference)
	replace mean_square_error = mean_square_error/ `df'
	//since the mean squared error is practically 0 then we are good
  
*****************************************************************************
/*
ii) Now, assume you are given the noisySine.csv as a time series with the values of 
xvalsSine.csv being the time variable. Filter the noisySine.csv data with any filter of your choice
and compare against cleanSine.csv to report the error.
*/
//Importing and merging the data for xvalsSine and cleanSine
	clear 

	import delimited "/san/RDS/Work/nielsen/Projects/Fatima/Atlanta_Fed/quant_spec_coding/data/noisySine.csv"
	rename v1 noisy_sine
	
	merge 1:1 _n using `x_and_clean_sine' //merging by observation count 
	drop _merge 
*****************************************************************************
	//generating a new variable with whole numbers 
	sort x 
	gen time_var = _n
	
	//setting time var as time variable 
	tsset time_var 
	
	//using a moving average as filter method 
	gen filtered_noisy_sine_x = (F1.noisy_sine + noisy_sine + L1.noisy_sine) / 3
	
	//calculating the mean squared error 
	gen squared_diff = (filtered_noisy_sine_x - sine_x) ^ 2
	sum squared_diff
	local mse = r(mean)
	display "Mean Squared Error: `mse'" 
exit 

	
	
