# FYP-2025
import pandas as pd
import numpy as np
from datetime import datetime
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Sales data - Order Delivered.csv', low_memory=False, parse_dates=['order_date', 'dispatch_date'])

# Data Preparation
df = df[(df['order_status'] == 'COMPLETED') & (df['gross_nmv'] > 0)]

# Calculate observation period (T)
current_date = df['order_date'].max() + pd.Timedelta(days=1)

# Calculate RFM features
rfm = df.groupby('store_id').agg({
    'order_date': [lambda x: (current_date - x.max()).days,  # Recency
                   lambda x: (current_date - x.min()).days], # T (age)
    'order_number': 'count',                                # Frequency
    'gross_nmv': 'sum'                                     # Monetary Value
})

# Flatten multi-index columns
rfm.columns = ['recency', 'T', 'frequency', 'monetary_value']

# Calculate average monetary value
rfm['avg_monetary'] = rfm['monetary_value'] / rfm['frequency']

# Filter for BG/NBD model (need frequency > 1)
rfm_bgnbd = rfm[rfm['frequency'] > 1].copy()
rfm_bgnbd['recency'] = np.minimum(rfm_bgnbd['recency'], rfm_bgnbd['T'])

# Split data
train, test = train_test_split(rfm_bgnbd, test_size=0.2, random_state=42)

# BG/NBD Model
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(train['frequency'], train['recency'], train['T'])

# Gamma-Gamma Model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(train['frequency'], train['avg_monetary'])

# Predict CLV 
test['predicted_clv'] = ggf.customer_lifetime_value(
    bgf,
    test['frequency'],
    test['recency'],
    test['T'],
    test['avg_monetary'], 
    time=90,
    freq='D'
)

# Evaluate models
#print("BG/NBD-GG Model Evaluation:")
#print(f"R-squared: {r2_score(test['monetary_value'], test['predicted_clv']/3)}")
#print(f"RMSE: {np.sqrt(mean_squared_error(test['monetary_value'], test['predicted_clv']/3))}")

# Prepare for ML models
ml_data = rfm.copy()
ml_data['target'] = ml_data['monetary_value']

# Split for ML
X = ml_data[['recency', 'frequency', 'avg_monetary']]
y = ml_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Evaluation:")
print(f"R-squared: {r2_score(y_test, rf_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred))}")

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

print("\nGradient Boosting Evaluation:")
print(f"R-squared: {r2_score(y_test, gb_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, gb_pred))}")

# Predict CLV for all customers
ml_data['predicted_clv'] = gb.predict(ml_data[['recency', 'frequency', 'avg_monetary']])

# Segergation function
def segment_customers(df):
    clv_75 = df['predicted_clv'].quantile(0.75)
    clv_25 = df['predicted_clv'].quantile(0.25)
    freq_median = df['frequency'].median()
    
    conditions = [
        (df['predicted_clv'] >= clv_75) & (df['frequency'] >= freq_median),
        (df['predicted_clv'] >= clv_75) & (df['frequency'] < freq_median),
        (df['predicted_clv'] < clv_75) & (df['predicted_clv'] >= clv_25) & (df['frequency'] >= freq_median),
        (df['predicted_clv'] < clv_25) | ((df['predicted_clv'] < clv_75) & (df['frequency'] < freq_median))
    ]
    
    segments = ['High Value Loyal', 'High Value At Risk', 'Low Value Loyal', 'Low Value Occasional']
    df['segment'] = np.select(conditions, segments)
    return df

ml_data = segment_customers(ml_data)

# Visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(data=ml_data, x='frequency', y='predicted_clv', hue='segment', palette='viridis')
plt.title('Customer Segmentation by CLV and Purchase Frequency')
plt.xlabel('Purchase Frequency')
plt.ylabel('Predicted CLV (3 months)')
plt.legend(title='Customer Segment')
plt.show()

# Segment statistics
segment_stats = ml_data.groupby('segment').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'predicted_clv': ['mean', 'count']
}).round(2)

print("\nSegment Statistics:")
print(segment_stats)

# Additional Visualizations for Customer Segments

# 1. Enhanced Scatter Plot with Marginal Distributions
plt.figure(figsize=(14, 10))
grid = sns.JointGrid(data=ml_data, x='frequency', y='predicted_clv', hue='segment', palette='viridis', height=10)
grid.plot_joint(sns.scatterplot, s=100, alpha=0.7)
grid.plot_marginals(sns.kdeplot, fill=True, alpha=0.5)
grid.ax_joint.set_xlabel('Purchase Frequency', fontsize=12)
grid.ax_joint.set_ylabel('Predicted CLV (3 months)', fontsize=12)
plt.suptitle('Customer Segmentation with Marginal Distributions', y=1.02, fontsize=14)
plt.legend(title='Customer Segment', bbox_to_anchor=(1.25, 1), borderaxespad=0)
plt.tight_layout()
plt.show()

# 2. Pie Chart for Segment Distribution
segment_counts = ml_data['segment'].value_counts()
plt.figure(figsize=(10, 8))
plt.pie(segment_counts, 
        labels=segment_counts.index, 
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('viridis', n_colors=4),
        explode=(0.05, 0, 0, 0),
        shadow=True)
plt.title('Customer Segment Distribution', fontsize=14)
plt.show()

# 3. Scatter Plot Faceted by Segment
plt.figure(figsize=(14, 10))
scatter = sns.relplot(
    data=ml_data,
    x='frequency',
    y='predicted_clv',
    col='segment',
    hue='segment',
    palette='viridis',
    col_wrap=2,
    height=5,
    aspect=1.5,
    s=100,
    alpha=0.7
)
scatter.set_axis_labels('Purchase Frequency', 'Predicted CLV (3 months)')
plt.suptitle('Customer Segments Breakdown', y=1.05, fontsize=14)
plt.tight_layout()
plt.show()

# 4. Donut Chart for Segment Value Contribution
segment_value = ml_data.groupby('segment')['predicted_clv'].sum()
plt.figure(figsize=(10, 8))
plt.pie(segment_value, 
        labels=segment_value.index, 
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('viridis', n_colors=4),
        wedgeprops={'width': 0.4},
        pctdistance=0.85)
centre_circle = plt.Circle((0,0), 0.3, color='white', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('CLV Value Contribution by Segment', fontsize=14)
plt.show()

# 5. Box Plots for Segment Comparison
plt.figure(figsize=(12, 8))
sns.boxplot(data=ml_data, x='segment', y='predicted_clv', palette='viridis')
plt.title('CLV Distribution Across Segments', fontsize=14)
plt.xlabel('Customer Segment', fontsize=12)
plt.ylabel('Predicted CLV (3 months)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\Recommendations:")
print("1. High Value Loyal: Premium loyalty programs, exclusive offers")
print("2. High Value At Risk: Reactivation campaigns, win-back offers")
print("3. Low Value Loyal: Upsell/cross-sell strategies, volume discounts")
print("4. Low Value Occasional: Cost-efficient engagement, evaluate ROI")

# Export the segmented data with predicted CLV
ml_data.to_csv('customer_segments_with_clv.csv', index=True)
print("Segmented data exported to 'customer_segments_with_clv.csv'")
