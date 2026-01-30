import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
LR = LogisticRegression(solver='liblinear')

# Load training data
data_df = pd.read_csv('heart.csv')

# Split features and target
x = data_df.drop( 'target', axis=1)
y = data_df['target']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train the model
LR.fit(x_train, y_train)

# Predict on test data
y_pred = LR.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy * 100)

# Load new test data
data_df = pd.read_csv('test_data.csv')

# Drop target column (if exists)
x = data_df.drop('target', axis=1)

# Predict new data
y_pred = LR.predict(x)

print(y_pred)
