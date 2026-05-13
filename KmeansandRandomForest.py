import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

path = "C:\\Users\\masan\\Downloads\\2026__28Candidate_29_Personal_Loans_Case_Dataset.xlsx"
df = pd.read_excel(path)
#print(df.head())
df = pd.get_dummies(df, columns=['Reason'], prefix='Reason')
df = pd.get_dummies(df, columns=['Fico_Score_group'], prefix='Fico')
df = pd.get_dummies(df, columns=['Employment_Status'], prefix='Emp')
df = pd.get_dummies(df, columns=['Employment_Sector'], prefix='EmpSec')
df = pd.get_dummies(df, columns=['Lender'], prefix='Lender')
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
print(df.head())
#create new dataframe with only numerical columns and remove Approved and all Lender columns
#numerical_df = df.select_dtypes(include=[np.number]).drop(columns=['Approved'] + [col for col in df.columns if col.startswith('Lender_')] + [col for col in df.columns if col.startswith('Reason_')]+ [col for col in df.columns if col.startswith('EmpSec_')])
#print(numerical_df.head())

#KMEANS CLUSTERING 
# Use the elbow method to determine the optimal number of clusters
#sse = []
#for k in range(1, 11):
   # kmeans = KMeans(n_clusters=k, random_state=42)
   # kmeans.fit(numerical_df)
  #  sse.append(kmeans.inertia_)
#plt.plot(range(1, 11), sse, marker='o')
#plt.title('Elbow Method For Optimal k')
#plt.xlabel('Number of clusters (k)')
#plt.ylabel('Sum of squared distances')
#plt.show()
# From the elbow plot, the optimal k is 4
#kmeans = KMeans(n_clusters=3, random_state=42)
#kmeans.fit(numerical_df)
#numerical_df['Cluster'] = kmeans.labels_
#print(numerical_df['Cluster'].value_counts())
# Visualize the clusters using a pairplot
#sns.pairplot(numerical_df, hue='Cluster', diag_kind='kde')
#plt.show()
# describe clusters
#for cluster in range(3):
 #   print(f"Cluster {cluster} description:")
  #  print(numerical_df[numerical_df['Cluster'] == cluster].describe())
#now figure out which clusters have the highest approval rates with which lenders
#df['Cluster'] = numerical_df['Cluster']
#approval_rates = df.groupby('Cluster')['Approved'].mean()
#print("Approval rates by cluster:")
#print(approval_rates)
#lender_approval = df.groupby('Cluster')[[col for col in df.columns if col.startswith('Lender_')]].mean()
#print("Lender approval rates by cluster:")
#print(lender_approval)

#RANDOM FOREST CLASSIFIER TO PREDICT APPROVAL
#make 3 new numerical dataframes one for lender A, one for lender B, one for lender C
lender_a_df = df[df['Lender_A'] == 1].drop(columns=['Lender_A', 'Lender_B', 'Lender_C', 'bounty'])
lender_b_df = df[df['Lender_B'] == 1].drop(columns=['Lender_A', 'Lender_B', 'Lender_C', 'bounty'])
lender_c_df = df[df['Lender_C'] == 1].drop(columns=['Lender_A', 'Lender_B', 'Lender_C', 'bounty'])
numerical_lender_a_df = lender_a_df.select_dtypes(include=[np.number])
numerical_lender_b_df = lender_b_df.select_dtypes(include=[np.number])
numerical_lender_c_df = lender_c_df.select_dtypes(include=[np.number])

def train_random_forest(df, lender_name):
    X = df.drop(columns=['Approved'])
    y = df['Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy for {lender_name}: {accuracy:.2f}")
    feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"Top 5 feature importances for {lender_name}:")
    print(feature_importances.head(5))
train_random_forest(numerical_lender_a_df, "Lender A")
train_random_forest(numerical_lender_b_df, "Lender B")
train_random_forest(numerical_lender_c_df, "Lender C")
#show feature importance for all three lenders in a single bar plot
importances_a = RandomForestClassifier(n_estimators=100, random_state=42).fit(
    numerical_lender_a_df.drop(columns=['Approved']),
    numerical_lender_a_df['Approved']
).feature_importances_
importances_b = RandomForestClassifier(n_estimators=100, random_state=42).fit(
    numerical_lender_b_df.drop(columns=['Approved']),
    numerical_lender_b_df['Approved']
).feature_importances_
importances_c = RandomForestClassifier(n_estimators=100, random_state=42).fit(
    numerical_lender_c_df.drop(columns=['Approved']),
    numerical_lender_c_df['Approved']
).feature_importances_
feature_names = numerical_lender_a_df.drop(columns=['Approved']).columns
x = np.arange(len(feature_names))
width = 0.25
plt.bar(x - width, importances_a, width, label='Lender A')
plt.bar(x, importances_b, width, label='Lender B')
plt.bar(x + width, importances_c, width, label='Lender C')
plt.xticks(x, feature_names, rotation=90)
plt.title('Feature Importances by Lender')
plt.legend()
plt.tight_layout()
plt.show()

