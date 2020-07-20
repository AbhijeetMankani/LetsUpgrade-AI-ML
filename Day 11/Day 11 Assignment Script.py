import pandas as pd

from scipy.stats import pearsonr

dataset = pd.read_csv("general_data.csv")

dataset = dataset.dropna()

dataset = dataset.drop_duplicates()


dataset.Attrition = dataset.Attrition.replace('Yes', 1)

dataset.Attrition = dataset.Attrition.replace('No', 0)

filtered_dataset = dataset[['Attrition', 'Age', 'DistanceFromHome', 'Education', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]

for i in filtered_dataset.columns:
    print(i, ':')
    print(pearsonr(filtered_dataset['Attrition'], filtered_dataset[i]))