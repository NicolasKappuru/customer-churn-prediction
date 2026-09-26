# customer-churn-prediction
Machine Learning for Customer Churn Prediction 

# Abstract
Customer Churn is a problem about customer discontinue a company's services or products. Machine learning techniques can be applied with purpose of predict if a customer 
has probability of be churn, this information can help the company to take decisions for not lost clients. This work search explore different machine learning techniques
to compare and get the best approach to solve this problem. 

# Methodology

This project uses the Telco Customer Churn dataset to establish supervised-learning baselines for predicting customer churn. The workflow is to acquire and validate the data, examine its structure, target distribution, and data quality, preprocess it into suitable feature representations, and compare traditional machine-learning models using consistent evaluation metrics. These results provide a reference for later approaches and help assess the effect of the class imbalance.

## Dataset Download and Integrity Verification

The dataset is available from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), published by BlastChar and identified there as originating from IBM Sample Data Sets. To reproduce the local setup:

1. Open the dataset page and review its usage terms and attribution requirements. The page does not specify a standard open-source license; the dataset is used here for academic learning and analysis, with attribution to its source and original authors.
2. Download the complete `WA_Fn-UseC_-Telco-Customer-Churn.csv` file. If it is provided in an archive, extract that CSV.
3. Place the file at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` in the repository.
4. Verify its SHA-256 checksum. The dataset provider does not supply a checksum; this value was calculated from the project copy as an integrity reference:

	```powershell
	Get-FileHash .\data\WA_Fn-UseC_-Telco-Customer-Churn.csv -Algorithm SHA256
	```

	Expected SHA-256: `88BE4B93FBE0CC83421AF1C503794C97C342ECA914C1576DB7C276E61D61358A`.

The CSV is approximately 977.5 KB (977,501 bytes) and contains 7,043 customer records with 21 columns: 20 customer/service features and the `Churn` target (`Yes` or `No`). The target contains 1,869 `Yes` and 5,174 `No` records, so the classes are imbalanced. The file is kept locally under `data/` and must be downloaded by each contributor.

# Repository Structure
Repository has 3 important directories in the distribution

**src:** has the code, is sectioned in the workshops specialization, like supervized learning, reinforcement learning, etc. 

**data:** contain the datasets, for that reason isn´t load in the remote repository, it´s specified in the env examples it use. 

**docs:** has the documentation of the project, divided in the 6 workshops using latex and libraries for the continue working by the team.  

