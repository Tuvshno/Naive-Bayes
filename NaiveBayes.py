from math import sqrt
from math import pi
from math import exp

# Calculate the mean of a list of numbers
def mean(numbers):
 return sum(numbers)/float(len(numbers))
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
 avg = mean(numbers)
 variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
 return sqrt(variance)

# Step 1 Seperate by class
# Seperate training data by class
def seperate_by_class(dataset):
  seperated = dict()

  for i in range(len(dataset)):
    vector = dataset[i]
    class_value = vector[-1]

    if(class_value not in seperated):
      seperated[class_value] = list()

    seperated[class_value].append(vector)
  
  return seperated

# Step 2 Summarize the dataset
# Calculate the mean, standard derivitive and count for each column
def summarize_dataset(dataset):
    summaries = []
    for column in zip(*dataset): #Converts the rows of dataset (*dataset) to column of tuples (zip)
        column_mean = mean(column)
        column_stdev = stdev(column)
        column_count = len(column)

        summaries.append((column_mean, column_stdev, column_count))

    del(summaries[-1]) #Removes the class variable
    return summaries

# Summarize data by class
def summarize_by_class(dataset):
    seperated = seperate_by_class(dataset)
    summaries = dict()
    for class_value, rows in seperated.items():
      summaries[class_value] = summarize_dataset(rows)
    
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
  exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
  return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calulculate the probability desnsies of the classes
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, count = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

