Fraud Detection Prediction App
Enter the transaction details below to predict whether a transaction is fraudulent or not.
Enter Transaction Information
This application evaluates transactions based on the following criteria:
Transaction Type: Type of the transaction (e.g., CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER). In this dataset, fraud occurs mainly in TRANSFER and CASH_OUT.
Transaction Amount: The value of the transaction.
Sender's Old Balance: The sender's balance before the transaction.
Receiver's Old Balance: The receiver's balance before the transaction.
Flagged by system rule: Whether the bank's rule-based system already flagged this transaction.
Note: If the transaction type is not CASH_IN and the amount is larger than the sender's old balance, this is unusual and can itself be a fraud signal.
Fraud Detection Result
The app will provide one of the following predictions:
YES! This Transaction is FRAUDULENT.
NO! This Transaction is NOT FRAUDULENT.
It also provides a Fraud Probability percentage and a risk level assessment:
High Risk (>= 70%): Recommend blocking / manual review.
Medium Risk (>= 30%): Recommend manual review.
Low Risk (< 30%).
You can also view the details used by the model, including the input summary and encoded features.
Disclaimer: This prediction is made by a machine-learning model and should support, not replace, human review.
