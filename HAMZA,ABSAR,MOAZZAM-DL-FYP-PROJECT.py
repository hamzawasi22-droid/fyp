# HAMZA ABSAR MOAZZAM
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import gradio as gr





df = pd.read_csv("D:/Churn_Modelling.csv")



df

# Dropping duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

#  Encoding categorical features
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Geography'] = le.fit_transform(df['Geography'])

#  Feature Selection
X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = df['Exited']

#  Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Distribution of Age by Exit status
sns.histplot(data=df, x='Age', hue='Exited', kde=True, bins=30)
plt.title("Age Distribution by Exit Status")
plt.show()


plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include='number')  # exclude strings like 'Hargrave'
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features Only)")
plt.show()



sns.countplot(x='Exited', data=df)
plt.title("Class Distribution: Exited vs Not Exited")
plt.xlabel("Exited (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()


sns.boxplot(x='Exited', y='EstimatedSalary', data=df)
plt.title("Salary vs Exit")
plt.xlabel("Exited")
plt.ylabel("Estimated Salary")
plt.show()


sns.boxplot(x='Exited', y='Balance', data=df)
plt.title("Balance vs Exit")
plt.xlabel("Exited")
plt.ylabel("Balance")
plt.show()


sns.countplot(data=df, x='Gender', hue=df['Exited'].astype(str))
plt.title("Gender vs Exit")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title='Exited')
plt.show()



sns.kdeplot(data=df[df['Exited']==0]['CreditScore'], label="Stayed", fill=True)
sns.kdeplot(data=df[df['Exited']==1]['CreditScore'], label="Exited", fill=True)
plt.title("Credit Score Distribution")
plt.legend()
plt.show()



selected_features = ['Age', 'Balance', 'CreditScore', 'EstimatedSalary', 'Exited']
sns.pairplot(df[selected_features], hue='Exited')
plt.suptitle("Pairplot of Key Features by Exit Status", y=1.02)
plt.show()


sns.violinplot(x="Gender", y="Age", hue="Exited", data=df, split=True)
plt.title("Age Distribution by Gender and Exit")
plt.show()


sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Data Heatmap")
plt.show()


# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))




# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))



# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, activation='relu', random_state=42, verbose=True)
mlp.fit(X_train, y_train)

# Predictions and accuracy
y_pred_mlp = mlp.predict(X_test)
print("MLP Classifier Report:")
print(classification_report(y_test, y_pred_mlp))

# Plot training loss
plt.plot(mlp.loss_curve_)
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


import gradio as gr

# Mapping used during LabelEncoding in preprocessing
geography_map = {v: i for i, v in enumerate(le.classes_) if 'Geography' in df.columns}
gender_map = {v: i for i, v in enumerate(le.classes_) if 'Gender' in df.columns}

def predict_exit(CreditScore, Geography, Gender, Age, Tenure, Balance,
                 NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    
    try:
        # Convert categorical values to encoded integers using pre-trained label encoders
        geo_val = geography_map.get(Geography)
        gender_val = gender_map.get(Gender)

        if geo_val is None or gender_val is None:
            return "Invalid input: unseen Geography or Gender"

        # Create feature array
        features = np.array([[CreditScore, geo_val, gender_val, Age, Tenure, Balance,
                              NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
        
        # Scale the input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = xgb.predict(features_scaled)[0]
        return "Will Exit" if prediction == 1 else "Will Stay"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Build Gradio interface
interface = gr.Interface(
    fn=predict_exit,
    inputs=[
        gr.Slider(300, 900, step=1, label="Credit Score"),
        gr.Dropdown(list(geography_map.keys()), label="Geography"),
        gr.Dropdown(list(gender_map.keys()), label="Gender"),
        gr.Slider(18, 92, step=1, label="Age"),
        gr.Slider(0, 10, step=1, label="Tenure"),
        gr.Number(label="Balance"),
        gr.Slider(1, 4, step=1, label="Num of Products"),
        gr.Radio([0, 1], label="Has Credit Card"),
        gr.Radio([0, 1], label="Is Active Member"),
        gr.Number(label="Estimated Salary")
    ],
    outputs="text",
    title="Bank Customer Exit Predictor",
    description="Predict if a customer will exit based on financial features"
)

interface.launch()





