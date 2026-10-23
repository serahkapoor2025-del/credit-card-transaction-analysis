# credit card fraud analysis

Small project I did to practice SQL and scikit-learn on a real dataset. I used the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, which has 284,807 transactions from European cardholders. Only about 0.17% of them are fraud, so most of the project ended up being about working with really imbalanced data.

Everything is in `credit_card_transaction_analysis.ipynb` (ran it in Google Colab).

## how it went

The data was pretty clean already, no missing values, but there were 1,081 duplicate rows so I dropped those.

The `Time` column is just seconds since the first transaction, so I turned it into an "hour" column (hour 0 to 23). Heads up, this is relative to the start of the data and not an actual clock time, so I wouldn't read too much into "morning" vs "night". I loaded it into SQLite and wrote a few queries (group by, order by, and a join to a small table I made that maps hours to 4 time periods).

Fraud rate by period:

| hours | transactions | fraud | rate |
|---|---|---|---|
| 0-5 | 23,842 | 115 | 0.482% |
| 6-11 | 70,643 | 118 | 0.167% |
| 12-17 | 96,121 | 133 | 0.138% |
| 18-23 | 93,120 | 107 | 0.115% |

The first six hours stood out, with a fraud rate roughly 3 to 4 times higher than the rest of the day. That period also has the fewest transactions, which probably helps explain it, but I didn't dig further.

## the model

I trained a basic logistic regression with `class_weight="balanced"` and a stratified 80/20 split. On the test set (56,746 transactions, 95 of them fraud):

- recall for fraud: 0.87
- precision for fraud: 0.06

So it catches most of the fraud but also flags a ton of normal transactions. That's what happens when you balance the weights this hard, and it's not something you could use in real life as is.

## stuff I'd fix

- I scaled `Amount` before splitting into train and test, which leaks a tiny bit of info. Should have done it after.
- Only tried one model. Would like to try tuning the threshold and look at a precision-recall curve.
- V1 to V28 are anonymized (PCA), so I couldn't really explain what drives the predictions.
