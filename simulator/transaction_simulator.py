import random
import time

def generate_transaction():
    transaction_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]

    tx = {
        "step": random.randint(1, 24),
        "type": random.choice(transaction_types),
        "amount": round(random.uniform(10, 10000), 2),
        "oldbalanceOrg": round(random.uniform(0, 20000), 2),
        "newbalanceOrig": round(random.uniform(0, 20000), 2),
        "oldbalanceDest": round(random.uniform(0, 20000), 2),
        "newbalanceDest": round(random.uniform(0, 20000), 2),
    }

    return tx
