# US Opioid Prescription 

## Background
The number of overdose deaths from substance abuse in the US was more than 56,000 in 2016, 
up from 33,000 in 2015. Prescription opioid abuse, misuse, and dependence as a public health hazard 
is a daily phenomenon now; according to Centers for Disease Control and Prevention (CDC), over 1,000 
people are treated in emergency department every day for misusing prescription opioid drugs. 

The major source of diverted opioids is physician prescription. However, opioids prescription to patients with acute pain and patients with chronic pain requires a careful distinction. As Opioid is regarded as one of the most effective drugs for the acute pain management, limiting its use for patients who are in urgent need of pain control, post surgical status, near end of life cancer patients, and other health crisis would not only be inhumane but also defeat its intended purpose. On the other hand, use of opioids for chronic non-malignant pain (CNMP) control has remained controversial for decades, and requires a closer look in regards to the current opioids health crisis. 
This study attempted to build a predictive model of the likelihood of a healthcare provider prescribing opioids drugs to patients with chronic pain.  More specifically, we used a variety of classification techniques to predict whether or not a  health care provider dispense prescription opioids in long term (more than 84 days). 



![Alt text](https://github.com/JennyLeeStat/Opioid/blob/master/assets/overdose_per_capita.png)

## Datasets

Two datasets, detailed data and provider summary, were obtained from the web page of
 the [Centers for Medicare and Medicaid Serices (CMS)](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/PartD2015.html).  
## Methods 

Five supervised classifiers candidates 1) stochastic gradient descent algorithm 
with logistic regression loss function, 2) stochastic gradient descent with linear support vector 
machine loss function, 3) perceptron, 4) naive Bayes multinomial classifier, 
5) Passive-aggressive classifier support partial_fit() function in scikit learn API. 
We compared the runtime, validation accuracy, and validation F-score at its default settings 
to select the best classifiers. 

Then we extended our analysis to deep neural network to test if deep learning algorithms 
improve the performance.


## Results
| Classifier | Test accuracy | Test F-beta(0.5)|
| ------------- | ------------- |-------------|
| SGD-Log  | 0.8751  | 0.6661|
| DNN  | 0.8782  | 0.6711|


## Discussion
![Alt text](https://github.com/JennyLeeStat/Opioid/blob/88256f0a2507a3c4d19664be77596cca6c128ec3/assets/estimated_prob.png)

Many features came up for both models. Carisoprodol, diazepam, alprazolam, gabapentin,  pregabalin, hydroxychloroquine sulfate, cyclobenzaprine HCl, and tizanidine HCl were highly associated with the label based on both model. As we expected from the EDA, of medical specialties, pain management, interventional pain management, rheumatology, hematology, and hospice and palliative care were estimated the label is true with higher probability for both model. Among states, only Alaska came up within 25 for both models. 



A limitation of the study is that the dataset only includes a snapshot of the total population as beneficiaries of Medicare are elderly population of age greater than 65. 