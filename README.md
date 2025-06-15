# Bulldozer Price Prediction


This project predicts auction prices for bulldozers and heavy equipment using machine learning. The solution includes comprehensive data cleaning, feature engineering, model comparison, and price prediction.

- Develop a reliable pricing model based on:
  - Equipment usage history
  - Machine specifications
  - Configuration details
- Create a "Blue Book" valuation guide for customers

The dataset contains:
- `Train.csv`: Historical auction data with prices (401,125 rows)
- `Valid.csv`: Validation set without prices (11,573 rows)
- `Test.csv`: Final test set for predictions (12,457 rows)

Key features:
- Equipment specifications (YearMade, ProductSize, etc.)
- Usage metrics (MachineHoursCurrentMeter)
- Auction details (saledate, state)

### Data Processing
- **NaN Handling**:
  - Dates: Coerced errors and filled median values
  - Numerics: Median imputation
  - Categoricals: 'missing' placeholder + 'unknown' category
- **Feature Engineering**:
  - Extracted `sale_year` and `sale_month` from dates
  - Log-transformed target variable (SalePrice)

### Model Comparison
Tested three algorithms:
1. Random Forest Regressor
2. Gradient Boosting Regressor
3. Ridge Regression

Evaluation metrics:
- **RMSLE** (Root Mean Squared Logarithmic Error)
- **MAE** (Mean Absolute Error in USD)
