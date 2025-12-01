# AI Cheat Detection in Online Games

This project demonstrates how machine learning can be applied to detect cheating behavior in online multiplayer games. It includes a Python script that generates a synthetic dataset, trains a classification model, evaluates its performance, and visualizes results. A presentation is also included to explain the system design and methodology.

## Repository Structure

```
├── cheat_detection.py
├── AI_Cheat_Detection_Presentation.pptx
├── README.md
```

## Project Overview

The goal of this project is to simulate an AI-based cheat detection system using gameplay telemetry. The model classifies player behavior as either legitimate or cheating based on features commonly associated with suspicious activity, such as abnormal accuracy, reaction time, and input patterns.

The project demonstrates:

* Dataset generation
* Model training using a Random Forest classifier
* Evaluation of the model with standard metrics
* Visualization of results using a confusion matrix
* Practical application of AI methods to game security

## Dependencies

The project requires the following Python packages:

* numpy
* pandas
* scikit-learn
* matplotlib

Install the dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## How to Run the Project

### 1. Navigate to the Project Directory

If cloning from a repository:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Run the Script

Execute the Python file:

```bash
python cheat_detection.py
```

## What the Script Does

### 1. Dataset Generation

The script creates a synthetic dataset containing features commonly used in cheat detection, including:

* actions_per_minute
* accuracy
* reaction_time
* movement_variation
* suspicious_reports

Each data point is labeled:

* 0 for a legitimate player
* 1 for a cheater

### 2. Model Training

A Random Forest classifier is trained to distinguish between legitimate and cheating behavior. The model configuration includes:

* 200 trees
* Maximum depth of 12
* Stratified train-test split

### 3. Model Evaluation

The script outputs:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification report
* Confusion matrix visualization

These metrics demonstrate how well the model identifies cheating behaviors based on the generated data.

### 4. Output Samples

The script displays a small table comparing the model's predictions with actual labels for the first 20 samples in the test set.

## Presentation Files

The included PowerPoint presentation explains:

* The project goals
* System description and dataset
* Pipeline and methodology
* Model performance
* Challenges and future improvements

These are intended for academic demonstration.

## Future Improvements

Potential enhancements include:

* Using real gameplay telemetry instead of synthetic data
* Expanding the model to sequence-based data
* Adding more features to improve model realism
* Evaluating more advanced machine learning models

## License

This project is intended for educational use. Add a license here if necessary.
