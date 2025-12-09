import pandas as pd
from faker import Faker
import random
import datetime

fake = Faker()

class BMWDataGenerator:
    def __init__(self, num_customers=1000):
        self.num_customers = num_customers
        self.customers = []
        self.interactions = []
        self.bmw_models = [
            {'model': '3 Series', 'fuel': 'Benzene', 'hp': 255, 'type': 'Sedan'},
            {'model': '5 Series', 'fuel': 'Diesel', 'hp': 288, 'type': 'Sedan'},
            {'model': '7 Series', 'fuel': 'Benzene', 'hp': 320, 'type': 'Sedan'},
            {'model': 'X3', 'fuel': 'Diesel', 'hp': 248, 'type': 'SUV'},
            {'model': 'X5', 'fuel': 'Benzene', 'hp': 335, 'type': 'SUV'},
            {'model': 'M3', 'fuel': 'Benzene', 'hp': 473, 'type': 'Sport'},
            {'model': 'M5', 'fuel': 'Benzene', 'hp': 600, 'type': 'Sport'},
            {'model': 'i4', 'fuel': 'Electric', 'hp': 281, 'type': 'Electric'},
            {'model': 'iX', 'fuel': 'Electric', 'hp': 322, 'type': 'Electric'}
        ]

    def generate_customers(self):
        print(f"Generating {self.num_customers} customers...")
        for _ in range(self.num_customers):
            gender = random.choice(['Male', 'Female'])
            first_name = fake.first_name_male() if gender == 'Male' else fake.first_name_female()
            last_name = fake.last_name()

            customer = {
                'customer_id': fake.uuid4(),
                'name': f"{first_name} {last_name}",
                'gender': gender,
                'age': random.randint(18, 80),
                'marital_status': random.choice(['Single', 'Married', 'Divorced', 'Widowed']),
                'has_kids': random.choice([True, False]),
                'social_activities': random.choice(['Sports', 'Travel', 'Reading', 'Music', 'Charity', 'None']),
                'owns_car': random.choice([True, False]),
                'job': fake.job(),
                'salary': random.randint(30000, 200000),
                'phone': fake.phone_number(),
                'home_address': fake.address().replace('\n', ', '),
                'work_address': fake.address().replace('\n', ', ')
            }
            self.customers.append(customer)

        return pd.DataFrame(self.customers)

    def generate_interactions(self, customers_df):
        print("Generating interactions...")
        interactions = []
        start_date = datetime.date.today() - datetime.timedelta(days=365*10)

        for _, customer in customers_df.iterrows():
            # Simulate number of visits per customer (0 to 10)
            num_visits = random.choices(range(0, 11), weights=[1, 2, 3, 2, 1, 0.5, 0.2, 0.1, 0.1, 0.05, 0.05])[0]

            for _ in range(num_visits):
                visit_date = fake.date_between(start_date=start_date, end_date='today')
                visit_type = random.choices(['Buy', 'Renew', 'Accessories', 'Look'], weights=[0.1, 0.1, 0.2, 0.6])[0]

                interaction = {
                    'customer_id': customer['customer_id'],
                    'visit_date': visit_date,
                    'visit_type': visit_type,
                    'car_model': None,
                    'car_year': None,
                    'car_fuel': None,
                    'car_hp': None,
                    'car_color': None,
                    'car_condition': None
                }

                if visit_type in ['Buy', 'Renew']:
                    car = random.choice(self.bmw_models)
                    interaction['car_model'] = car['model']
                    interaction['car_year'] = random.randint(2015, 2024)
                    interaction['car_fuel'] = car['fuel']
                    interaction['car_hp'] = car['hp']
                    interaction['car_color'] = fake.color_name()
                    interaction['car_condition'] = random.choice(['New', 'Used'])

                # Combine customer data with interaction data
                full_record = {**customer.to_dict(), **interaction}
                interactions.append(full_record)

        self.interactions_df = pd.DataFrame(interactions)
        return self.interactions_df

if __name__ == "__main__":
    generator = BMWDataGenerator(num_customers=500)
    customers = generator.generate_customers()
    data = generator.generate_interactions(customers)

    # Save to CSV
    data.to_csv('bmw_customer_data.csv', index=False)
    print(f"Generated {len(data)} records and saved to 'bmw_customer_data.csv'.")
