#Author: Michael San Pietro
#Date: 1/20/2026

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

path = "C:\\Users\\masan\\Downloads\\2026__28Candidate_29_Personal_Loans_Case_Dataset.xlsx"
df = pd.read_excel(path)

#bounty by lender 
lenderA_bounty = 150
lenderB_bounty = 250
lenderC_bounty = 350

#create a new dataframe with only the following columns: 'Loan_Amount', 'FICO_Score', Employment_Status(split into dummies), Monthly_Gross_Income, Monthly_Housing_Payment, Ever_Bankrupt_or_Foreclose, Approved
relevant_df = df[['Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose', 'Approved', 'Lender', 'Employment_Status']]
relevant_df = pd.get_dummies(relevant_df, columns=['Employment_Status'], prefix='Emp')
#change dummies to numeric
for col in relevant_df.columns:
    if relevant_df[col].dtype == 'bool':
        relevant_df[col] = relevant_df[col].astype(int)
#drop unemployment to avoid perfect multicollinearity
relevant_df = relevant_df.drop(columns=['Emp_unemployed'])

#Designing an algorithm to measure prospective probability using the following procedure:
#Create Logistic Regression Models in Python for each Lender based on our determined statistically significant criteria
#For each row, use the coefficients to determine the probability of acceptance for each lender
#Multiply that probability times the prospective bounty
#Assign the row to the highest lender for that output

import statsmodels.api as sm
def logistic_regression_model(data):
    X = data.drop(columns=['Approved'])
    y = data['Approved']
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0)
    return model
lender_a_df = relevant_df[relevant_df['Lender'] == 'A'].drop(columns=['Lender'])
lender_b_df = relevant_df[relevant_df['Lender'] == 'B'].drop(columns=['Lender'])
lender_c_df = relevant_df[relevant_df['Lender'] == 'C'].drop(columns=['Lender'])
print(lender_a_df.head())
#now run these to make sure they work
model_a = logistic_regression_model(lender_a_df)
#now save the coefficients for each model
model_b = logistic_regression_model(lender_b_df)
model_c = logistic_regression_model(lender_c_df)
coeffs_a = model_a.params
coeffs_b = model_b.params
coeffs_c = model_c.params
#now using these coefficients, calculate the prospective probability for each lender for each row in relevant_df
def calculate_probability(row, coeffs):
    z = coeffs['const']
    for var in coeffs.index:
        if var != 'const':
            z += coeffs[var] * row[var]
    prob = 1 / (1 + np.exp(-z))
    return prob

#for each row in relevant_df, calculate the probability for each lender and multiply by bounty
def assign_lender(row):
    prob_a = calculate_probability(row, coeffs_a) * lenderA_bounty
    prob_b = calculate_probability(row, coeffs_b) * lenderB_bounty
    prob_c = calculate_probability(row, coeffs_c) * lenderC_bounty
    if prob_a >= prob_b and prob_a >= prob_c:
        return 'A', prob_a
    elif prob_b >= prob_a and prob_b >= prob_c:
        return 'B', prob_b
    else:
        return 'C', prob_c
relevant_df[['Assigned_Lender', 'Expected_Bounty']] = relevant_df.apply(assign_lender, axis=1, result_type='expand')
print(relevant_df[['Assigned_Lender', 'Expected_Bounty']].head())
# for each row, check if the assigned lender matches the actual, if it doesn't then add the expected bounty to a total
total_additional_revenue = 0
current_total_bounty = 0
for index, row in relevant_df.iterrows():
    if row['Lender'] == 'A':
        current_total_bounty += lenderA_bounty if row['Approved'] else 0
    elif row['Lender'] == 'B':
        current_total_bounty += lenderB_bounty if row['Approved'] else 0
    elif row['Lender'] == 'C':
        current_total_bounty += lenderC_bounty if row['Approved'] else 0
    if row['Assigned_Lender'] != row['Lender']:
        if row['Approved']==1:
            #subtract the current bounty from the new expected bounty to get additional revenue
            if row['Assigned_Lender'] == 'A':
                total_additional_revenue += row['Expected_Bounty'] - lenderA_bounty
            elif row['Assigned_Lender'] == 'B':
                total_additional_revenue += row['Expected_Bounty'] - lenderB_bounty
            elif row['Assigned_Lender'] == 'C':
                total_additional_revenue += row['Expected_Bounty'] - lenderC_bounty
        else:
            total_additional_revenue += row['Expected_Bounty'] 
print(f"Current Total Bounty: ${current_total_bounty}")
print(f"Total Additional Revenue from Reassignment: ${total_additional_revenue}")
#export relevant_df to excel
relevant_df.to_excel("C:\\Users\\masan\\Downloads\\2026__28Candidate_29_Personal_Loans_Case_Dataset_With_Assignments.xlsx", index=False)














