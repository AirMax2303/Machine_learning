import pandas as pd

# Task 1: Upload data
data = pd.read_csv("math_students.csv")

# Task 2: Print the first and last 10 rows of the table
print("First 10 rows:")
print(data.head(10))
print("\nLast 10 rows:")
print(data.tail(10))

# Task 3: Output the number of objects and their characteristics
print("\nNumber of objects and their characteristics:")
print(data.shape)

# Task 4: Output the names of all columns
print("\nColumn names:")
print(data.columns)

# Task 5: Are there any gaps in the data
print("\nCheck for missing values:")
print(data.isnull().any().any())

# Task 6: Output statistics on the values of the signs
print("\nStatistics on the values of the features:")
print(data.describe())

# Task 7: Output a more detailed description of the feature values
print("\nDetailed description of feature values:")
print(data.info())

# Task 8: What values does one of the signs take (for example, Fjob)?
print("\nUnique values of Fjob:")
print(data['Fjob'].unique())

# Task 9: Withdraw only those students whose guardian is the father and works as a doctor or engineer
selected_students = data[(data['guardian'] == 'father') & (data['Fjob'].isin(['doctor', 'engineer']))]
print("\nStudents whose guardian is the father and works as a doctor or engineer:")
print(selected_students)

# Task 10: Create a "study_time_ratio" attribute
data['study_time_ratio'] = data['studytime'] / (data['studytime'] + data['freetime'])

# Task 11: Display new size and new columns
print("\nSize and new columns:")
print(data.shape)
print(data.columns)

# Task 12: Display the most common number of unreleased items
print("\nMost common number of unreleased items:")
print(data['failures'].mode().iloc[0])

# Task 13: Find the number of students whose mother and father work
working_parents_count = data[(data['Mjob'].isin(['teacher', 'health', 'services', 'at_home', 'other']))
                            & (data['Fjob'].isin(['teacher', 'health', 'services', 'at_home', 'other']))].shape[0]
print("\nNumber of students whose mother and father work:", working_parents_count)

# Task 14: Find the maximum age of students whose both parents work in the service sector (police)
max_age = data[(data['Mjob'].isin(['services', 'police'])) & (data['Fjob'].isin(['services', 'police']))]['age'].max()
print("\nMaximum age of students whose both parents work in the service sector (police):", max_age)

# Task 15: Find the number of students who have a grade for the first semester above the average score
above_average_count = data[data['G1'] > data['G1'].mean()].shape[0]
print("\nNumber of students with a grade for the first semester above the average score:", above_average_count)

# Task 16: Divide the students into two groups based on mothers' education and compare average final scores
grouped_data = data.groupby('Medu')['G3'].mean()
print("\nAverage final scores based on mothers' education:")
print(grouped_data)
