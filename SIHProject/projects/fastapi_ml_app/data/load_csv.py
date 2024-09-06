import pandas as pd

# Path to your CSV file
file_path = '../data/SIH Data.csv'

# Load the data into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())

