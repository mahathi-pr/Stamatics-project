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


# Dog Breed Classification using EfficientNet-B3 

This project classifies dog breeds based on image data using a deep learning model. The solution includes robust data preprocessing, augmentation, transfer learning with EfficientNet-B3, and evaluation using standard metrics.

### Objective:
Build a high-accuracy dog breed classifier from scratch using:
- **Labeled dog images**
- **Pretrained convolutional neural networks (EfficientNet-B3)**
- **Weighted loss for class imbalance**


### Dataset
Kaggle competition: [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

- `labels.csv`: Train image IDs with breed names (10,222 samples)
- `train/`: Folder with training images
- `test/`: Folder with test images for submission
- `sample_submission.csv`: Submission format

### Data Processing
- **Label Encoding**: Breeds encoded into integer class labels
- **Stratified Split**: 90% train / 10% validation
- **Augmentations**:
  - Resize (300x300), RandomHorizontalFlip, Rotation, ColorJitter
- **Normalization**: Applied using ImageNet statistics

### Model Architecture
- **Base Model**: EfficientNet-B3 (pretrained)
- **Final Layer**: Fully connected → softmax over 120 breeds
- **Loss Function**: `CrossEntropyLoss` with label smoothing and class weights
- **Optimizer**: Adam with learning rate scheduling
- **Early Stopping**: Based on validation accuracy

###Evaluation Metrics
- **Accuracy** (top-1 classification)
- **Weighted F1-Score** for multi-class performance tracking.
