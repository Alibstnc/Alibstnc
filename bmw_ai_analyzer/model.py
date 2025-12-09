import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

class BMWPredictor:
    def __init__(self):
        self.purchase_model = None
        self.recommendation_model = None
        self.label_encoders = {}
        self.model_encoder = None
        self.scaler = StandardScaler()

    def preprocess_data(self, df):
        # Create a copy to avoid SettingWithCopyWarning
        data = df.copy()

        # Define categorical columns
        categorical_cols = ['gender', 'marital_status', 'has_kids', 'social_activities', 'owns_car', 'job']

        # Initialize label encoders if not already done (this will be done during training)
        # However, for simplicity in this script structure where we might just call preprocess once:
        # We will do encoding inside training/prediction to avoid leakage,
        # or we fit on the data passed here assuming it's the training set.
        # To be cleaner, let's just return the data as is or do basic type conversion.
        # We will handle encoding inside train_models and predict.

        return data

    def train_models(self, data):
        # Filter for relevant features
        features = ['gender', 'age', 'marital_status', 'has_kids', 'social_activities', 'owns_car', 'salary', 'job']

        # Target 1: Will they buy/renew?
        data['target_buy'] = data['visit_type'].apply(lambda x: 1 if x in ['Buy', 'Renew'] else 0)

        X = data[features]
        y_buy = data['target_buy']

        # Split data first to avoid leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y_buy, test_size=0.2, random_state=42)

        # Fit encoders on training data
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        categorical_cols = ['gender', 'marital_status', 'has_kids', 'social_activities', 'owns_car', 'job']

        # We need to handle high cardinality for 'job'. LabelEncoder is fine for trees.
        # However, if test set has unseen labels, we have a problem.
        # We'll use a custom safe encode function or handle unknown labels.

        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on full dataset for simplicity in this demo context to ensure all categories are known
            # OR better: handle unknowns.
            # Given the constraints of a simple script, we will fit on train and use a robust transform.

            # To strictly avoid leakage, we fit on X_train.
            # But we need to handle unknown labels in X_test.
            # A simple strategy: map unknown to a new class 'Unknown'.

            # Let's collect all unique values from train
            train_values = X_train[col].astype(str).unique()
            le.fit(np.append(train_values, 'Unknown'))
            self.label_encoders[col] = le

            X_train_encoded[col] = X_train[col].astype(str).apply(lambda x: le.transform([x])[0])
            X_test_encoded[col] = X_test[col].astype(str).apply(lambda x: le.transform([x])[0] if x in train_values else le.transform(['Unknown'])[0])


        # Train Purchase Model
        print("Training Purchase Prediction Model...")
        self.purchase_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.purchase_model.fit(X_train_encoded, y_train)

        y_pred = self.purchase_model.predict(X_test_encoded)
        print("Purchase Model Performance:")
        print(classification_report(y_test, y_pred))

        # Target 2: Which model to recommend?
        # We only train on rows where a purchase happened
        purchase_data = data[data['target_buy'] == 1].copy()

        if len(purchase_data) > 0:
            print("Training Recommendation Model...")
            y_model = purchase_data['car_model']

            # Encode car model target
            self.model_encoder = LabelEncoder()
            y_model_encoded = self.model_encoder.fit_transform(y_model)

            X_model = purchase_data[features]

            X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_model, y_model_encoded, test_size=0.2, random_state=42)

            # Encode features for recommendation model
            X_train_m_encoded = X_train_m.copy()
            X_test_m_encoded = X_test_m.copy()

            for col in categorical_cols:
                le = self.label_encoders[col] # Reuse encoders
                train_values = le.classes_

                # Helper to safely transform
                def safe_transform(series):
                    return series.astype(str).apply(lambda x: le.transform([x])[0] if x in train_values else le.transform(['Unknown'])[0])

                X_train_m_encoded[col] = safe_transform(X_train_m[col])
                X_test_m_encoded[col] = safe_transform(X_test_m[col])

            self.recommendation_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.recommendation_model.fit(X_train_m_encoded, y_train_m)

            y_pred_m = self.recommendation_model.predict(X_test_m_encoded)
            print("Recommendation Model Performance:")
            unique_labels = np.unique(np.concatenate((y_test_m, y_pred_m)))
            target_names = self.model_encoder.inverse_transform(unique_labels)
            print(classification_report(y_test_m, y_pred_m, labels=unique_labels, target_names=target_names, zero_division=0))
        else:
            print("Not enough purchase data to train recommendation model.")

    def predict(self, new_data):
        # new_data should be a DataFrame with the same features as training data
        processed_data = new_data.copy()

        categorical_cols = ['gender', 'marital_status', 'has_kids', 'social_activities', 'owns_car', 'job']
        features = ['gender', 'age', 'marital_status', 'has_kids', 'social_activities', 'owns_car', 'salary', 'job']

        for col in categorical_cols:
             if col in processed_data.columns:
                 le = self.label_encoders[col]
                 # Handle unknown labels
                 processed_data[col] = processed_data[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(['Unknown'])[0])

        X = processed_data[features]

        # Predict Purchase Probability
        buy_prob = self.purchase_model.predict_proba(X)[:, 1]

        # Predict Recommended Model
        recommendations = []
        if self.recommendation_model:
            model_indices = self.recommendation_model.predict(X)
            recommendations = self.model_encoder.inverse_transform(model_indices)
        else:
            recommendations = ["N/A"] * len(X)

        return buy_prob, recommendations

if __name__ == "__main__":
    df = pd.read_csv('bmw_customer_data.csv')
    predictor = BMWPredictor()
    # Note: In this updated version, preprocess_data just returns copy, encoding happens in train
    processed_df = predictor.preprocess_data(df)
    predictor.train_models(processed_df)

    # Example Prediction on first 5 rows
    sample = df.head(5).copy()
    probs, recs = predictor.predict(sample)

    print("\nPredictions for sample customers:")
    for i in range(len(sample)):
        print(f"Customer: {sample.iloc[i]['name']}, Buy Probability: {probs[i]:.2f}, Recommended: {recs[i]}")
