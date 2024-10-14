# project-2-naturalistic-driving-app-data

This is a machine learning project for traffic data analysis aimed at predicting vehicle behavior near intersections.

## Project Overview

This project uses machine learning models to analyze and predict changes in vehicle speed, acceleration, and angle near intersections. It employs various data processing and visualization techniques to help understand vehicle behavior.

## Features

- Analyze changes in vehicle speed, acceleration, and angles
- Predict vehicle stopping probabilities
- Visualize changes in vehicle acceleration, speed, and angular velocity over time
- Support for multiple data input formats

## Installation

1. Clone the project locally:：
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   ```
   
2. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:

   ```bash
   main.ipynb
   ```

## Usage

1. Load the dataset and run the analysis:：

   ```bash
   !python src/stages/data_load.py --config=params.yaml
   ```

2. Extract features:

   ```bash
   !python src/stages/featurize.py --config=params.yaml
   ```

3. Train the model and generate plots:

   ```bash
   !python src/stages/run_complete_pipeline1.py --config_path=params.yaml --save_dir=output
   ```

   

