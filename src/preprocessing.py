import pandas as pd
import numpy as np
from ingestion import load_csv, load_json

def clean_cards(cards):
    cards = cards.rename(columns={'id': 'card_id'})

    # sensitive information, high cardinality
    cards.drop(columns=['card_number', 'cvv'], inplace=True)

    # map yes and no to binary values; map for reusability
    yes_no_map = {'YES': 1, 'NO': 0}
    cards['has_chip'] = cards['has_chip'].str.upper().map(yes_no_map)
    cards['card_on_dark_web'] = cards['card_on_dark_web'].str.upper().map(yes_no_map)

    # convert credit limit to numeric, remove currency
    cards['credit_limit'] = (cards['credit_limit'].str.replace('$', '', regex=False).astype(float))

    # process time data
    DATE_FORMATS = {
        'expires': '%m/%Y',
        'acct_open_date': '%m/%Y',
        'year_pin_last_changed': '%Y'
    }

    for col, fmt in DATE_FORMATS.items():
        cards[col] = pd.to_datetime(cards[col], format=fmt, errors='coerce')

    return cards


def clean_users(users):
    users = users.rename(columns={'id': 'client_id'})

    # drop information not used later, see zip removal in transactions
    users.drop(columns=['address', 'latitude', 'longitude'], inplace=True)

    # map gender to binary variable  THIS CREATES BIASED MODEL, CHANGE AND MOVE TO FEATURE ENGINEERING SCRIPT
    '''
    if 'gender' in users.columns:
        clean_gender = users['gender'].astype(str).str.strip().str.lower()
        users['is_female'] = clean_gender.map({'female': 1}).fillna(0).astype(int)
        users = users.drop(columns=['gender'])
        '''
    
    # convert to numeric, replace currency value
    for col in ['per_capita_income', 'yearly_income', 'total_debt']:
        users[col] = users[col].str.replace('$', '').astype(float)

    return users


def clean_transactions(transactions):
    
    #convert to datetime
    transactions['date'] = pd.to_datetime(transactions['date'])
    
    #drop information not used later
    transactions.drop(columns=['merchant_id', 'merchant_city', 'merchant_state', 'zip', 'mcc'], inplace=True)

    # convert to numeric, replace currency value
    transactions['amount'] = pd.to_numeric(transactions['amount'].str.replace('$', '', regex=False), errors='coerce')
    transactions['errors'] = transactions['errors'].fillna('None')

    return transactions


def clean_labels(labels):
    #adjust format of labels for later merging with transactions
    labels = labels.T
    labels.reset_index(inplace=True)
    labels.rename(columns={'index': 'id'}, inplace=True)

    #convert id to numeric for later matching
    labels['id'] = labels['id'].astype('int64')

    #map target to binary
    labels['target'] = labels['target'].map({'Yes': 1, 'No': 0})
    
    return labels


if __name__ == "__main__":
    test_data = load_json("train_fraud_labels.json")
    clean_test_data = clean_labels(test_data)
    print(clean_test_data.head())