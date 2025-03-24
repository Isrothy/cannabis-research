# Cannabis App Classifier

This module provides tools to build and use a machine learning classifier that identifies cannabis-related mobile applications based on their classifier scores from various language models.

## Features

- Random Forest classifier with bootstrapping for robust predictions
- Class weighting to handle imbalanced data
- Sample weighting based on cluster weights
- Comprehensive evaluation metrics and visualizations
- Prediction tools for new applications

## Files

- `cannabis_app_classifier.py`: Main script to train and evaluate the classifier
- `predict_cannabis_app.py`: Script to use the trained model for predictions

## Requirements

- Python 3.6+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- pyyaml

## Usage

### Training the Classifier

```bash
python cannabis_app_classifier.py --apple_data data/appleAppsSamples-marked.csv --google_data data/googleAppsSamples-marked.csv --output_model models/cannabis_app_classifier.joblib
```

#### Options:

- `--apple_data`: Path to Apple apps CSV file (default: data/appleAppsSamples-marked.csv)
- `--google_data`: Path to Google apps CSV file (default: data/googleAppsSamples-marked.csv)
- `--output_model`: Path to save the trained model (default: models/cannabis_app_classifier.joblib)
- `--output_metrics`: Path to save evaluation metrics (default: models/cannabis_app_metrics.json)
- `--n_estimators`: Number of trees in the random forest (default: 100)
- `--max_depth`: Maximum depth of the trees (default: None)
- `--n_bootstrap`: Number of bootstrap samples (default: 5)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--random_state`: Random seed for reproducibility (default: 42)
- `--config`: Path to YAML configuration file
- `--plot_dir`: Directory to save plots (default: img)

### Making Predictions

```bash
python predict_cannabis_app.py --model models/cannabis_app_classifier.joblib --input new_apps.csv --output predictions.csv
```

#### Options:

- `--model`: Path to the trained model (default: models/cannabis_app_classifier.joblib)
- `--input`: Path to input file (CSV or JSON)
- `--output`: Path to save predictions
- `--format`: Input file format (csv or json, auto-detected if not specified)

## Input Data Format

The classifier expects the following features for each app:

- `facebook_bart-large-mnli`: Score from the facebook/bart-large-mnli model
- `roberta-large-mnli`: Score from the roberta-large-mnli model
- `distilbert-base-uncased-finetuned-sst-2-english`: Score from the distilbert model
- `valhalla_distilbart-mnli-12-9`: Score from the valhalla/distilbart-mnli-12-9 model

For training, the data should also include:

- `cannabis_related`: Binary label (1 for cannabis-related, 0 for not)
- `cluster_label`: Cluster assignment from clustering algorithm
- `cluster_weight`: Weight of the cluster

## Model Details

The classifier uses a Random Forest algorithm with the following enhancements:

1. **Bootstrapping**: Multiple models are trained on bootstrap samples and combined for prediction
2. **Class Weighting**: The `class_weight='balanced'` parameter adjusts weights inversely proportional to class frequencies
3. **Sample Weighting**: Each sample is weighted based on its cluster weight
4. **Feature Scaling**: StandardScaler is applied to normalize feature values

## Output

The training process produces:

1. A trained model file (.joblib)
2. Evaluation metrics (.json)
3. Visualization plots:
   - Feature importance
   - Confusion matrix
   - Precision-recall curve

The prediction process adds two columns to the input data:

- `predicted_cannabis`: Binary prediction (1 for cannabis-related, 0 for not)
- `cannabis_probability`: Probability of being cannabis-related (0-1)
