import pandas as pd
import sys
from data_generator import BMWDataGenerator
from model import BMWPredictor

def main():
    print("Welcome to the BMW AI Analyzer")

    # Step 1: Generate Data
    print("\n--- Step 1: Generating Synthetic Data ---")
    generator = BMWDataGenerator(num_customers=1000)
    customers = generator.generate_customers()
    data = generator.generate_interactions(customers)
    data.to_csv('bmw_customer_data.csv', index=False)
    print("Data generation complete.")

    # Step 2: Train Model
    print("\n--- Step 2: Training AI Models ---")
    predictor = BMWPredictor()
    processed_df = predictor.preprocess_data(data)
    predictor.train_models(processed_df)

    # Step 3: Make Predictions
    print("\n--- Step 3: Predicting Future Customer Behavior ---")
    # For prediction, we simulate "current" customers who might buy.
    # We can just pick a random sample of unique customers from the generated data
    # and predict their next move.

    unique_customers = data.drop_duplicates(subset=['customer_id']).copy()

    # We need to make sure the data format matches what the model expects.
    # The model expects the raw features which it then encodes.
    # We can pass the unique_customers dataframe directly as it has the raw features.

    buy_probs, recommendations = predictor.predict(unique_customers)

    unique_customers['buy_probability'] = buy_probs
    unique_customers['recommended_model'] = recommendations

    # Filter for high probability customers (e.g., > 0.3 just to show some results given low accuracy)
    potential_buyers = unique_customers[unique_customers['buy_probability'] > 0.3].sort_values(by='buy_probability', ascending=False)

    print(f"\nIdentified {len(potential_buyers)} potential buyers out of {len(unique_customers)} customers.")
    print("\nTop 10 High Potential Customers:")
    print(potential_buyers[['name', 'age', 'salary', 'buy_probability', 'recommended_model']].head(10).to_string(index=False))

    # Save predictions
    potential_buyers.to_csv('bmw_sales_predictions.csv', index=False)
    print("\nPredictions saved to 'bmw_sales_predictions.csv'.")

if __name__ == "__main__":
    main()
