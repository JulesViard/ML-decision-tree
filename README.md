# ** ML Algo Classifier :** Decision Trees

**Subject:** Implementing a Decision Tree algorithm from scratch to predict the position on a user in one of four rooms according to his detection of wifi power of the 7 emiters.

## A. Introduction

This Python script demonstrates how to train and evaluate a decision tree classifier. The code is structured to work with two different datasets: a "clean" dataset and a "noisy" dataset.
You can use this code to analyze the performance of a decision tree classifier on your own datasets by modifying the file_path of the clean and noisy data_set.
This README provides information on how to use the script, interpret the results, and modify it for your specific datasets.

**NB: Please ensure to have initialize both of the clean and noisy data-set file-paths before running the file.**

## B. Steps of usage

1. **Prerequisites**:
   - Python 3.x
   - Required Python libraries (numpy, matplotlib)

  ```bash
  pip install -r requirements.txt
  ```

2. **Directory Structure**:
   Ensure your directory structure is organized like this:

Path-to-your-folder
├── decision_tree_classifier.py
├── README.md
├── wifi_db
├──── clean_data.txt
├──── noisy_data.txt

You can replace `clean_data.txt` and `noisy_data.txt` with your dataset files.

**NB: In this case, please ensure to keep the same name for the files you are uploading**

3. **Data Preparation**:

- Make sure your dataset files are in .txt format with the same structure (file name, attributes and labels).

4. **Running the Script**:
- Open a terminal and navigate to the script's directory.
- Execute the script using the following command:

   ```bash
   python decision_tree_classifier.py
   ```

5. **Results**:
- After running the script, it will produce a variety of results and metrics for the decision tree classifier on both clean and noisy datasets (firstly the results without pruning of both of the clean and noisy datasets, and after the results with pruning).
  The approximate time to run the entiere .py file is roughly 10 minutes.

## C. Understanding the Results

- The script will provide detailed results of a k-fold cross-validation (or a nested cross-validation) process for each datasets. This includes accuracy, precision, recall, F1-score, confusion matrices, class accuracies, and tree depth.
The mean and standard deviation of the metrics across all folds are also displayed, giving you an overall performance summary.

- The script distinguishes between the clean dataset and noisy dataset for both without pruning and with pruning scenarios. Each scenario's results are presented separately in the terminal.

## D. Modifying for Your Dataset

To use this script with your own dataset, follow these steps:

1. Replace `clean_data.csv` and `noisy_data.csv` in the script's directory with your dataset files.

2. You may need to modify other script parameters (e.g., random seed, number of folds) to suit your specific analysis.

3. Run the script as described in the "Usage" section to analyze your dataset with the decision tree classifier.

## E. Additional Notes

- You can save the confusion matrices and other metrics to files using the provided functions.

- A tree visualization could be created and displayed after the full execution of the script. You can replace the current tree (trained on the full clean dataset provided) with the one generated using your data. The plot tree function is commented.
