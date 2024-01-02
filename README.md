<!-- #region -->
# Predicting Organ Injuries from Blunt Force Trauma using CT Scans and Machine Learning. 

![image](https://github.com/ChrisRichardMiles/rsna_ab_trauma/blob/525007eeeae6ecf42c0d6629022cab0ac9fbe8db/Screenshot%202024-01-02%20023350.png)

## Problem: we need faster diagnoses of organ injuries
A recent study has estimated the incident rate for trauma patients with solid organ injuries at 5.4 per 100,000. Of these patients, 12.5% die within 30 days (Larsen, J. W., et al., 2022). It is difficult to determine if a patient with abdominal trauma has solid organ injuries that need immediate surgery. "Computed tomography (CT) has become an indispensable tool in evaluating patients with suspected abdominal injuries due to its ability to provide detailed cross-sectional images of the abdomen" (Errol Colak, et al., 2023). However, the process is time consuming, and requires experts to analyze the scans. 


## Deliverables
 * A model that takes a CT scan and produces probabilities for the injury status of a patients liver
 * A model that can segment out the liver from a CT scan
 * A report and slide deck outlining process, outcomes, and recommendations
 
## Stakeholders
Doctor and radiologists who use abdominal CT scans of patients and need to diagnose the injury status of their liver, especially when deciding if surgery is needed.


## Proposed solution
A two stage deep learning model that takes a CT scan as an input, and predicts the injury status of an organ. The predictions will be probabilities for each of three labels: no injury, mild injury, and severe injury. A severe injury is one which requires immediate surgery (Errol Colak, et al., 2023). The first stage of the model predicts the segmentation mask of the organ. The second stage of the model predicts the injury status of the organ. 

In this study I will focus on the liver, since it is the most common injury found in abdominal trauma patients (Larsen, J. W., et al., 2022). 
<!-- #endregion -->
