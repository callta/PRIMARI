# Benefit, Harm, and Cost-effectiveness Associated With Magnetic Resonance Imaging Before Biopsy vs Biopsy-First Screening For Prostate Cancer

## Citation
Callender T, Emberton M, Morris S, Pharoah PDP, Pashayan N. Benefit, Harm, and Cost-effectiveness Associated with Magnetic Resonance Imaging Before Biopsy in Age-based and Risk-stratified Screening for Prostate Cancer. JAMA Network Open. 2021;4(2):e2037657. doi:10.1001/jamanetworkopen.2020.37657 

## Abstract
**IMPORTANCE** If magnetic resonance imaging(MRI) mitigates overdiagnosis of prostate cancer while improving the detection of clinically significant cases, including MRI in a screening program for prostate cancer could be considered.

**OBJECTIVE** To evaluate the benefit-harm profiles and cost-effectiveness associated with MRI before biopsy compared with biopsy-first screening for prostate cancer using age-based and risk- stratified screening strategies.

**DESIGN, SETTING, AND PARTICIPANTS** This decision analytical model used a life-table approach and was conducted between December 2019 and July 2020. A hypothetical cohort of 4.48 million men in England aged 55 to 69 years were analyzed and followed-up to 90 years of age.

**EXPOSURES** No screening, age-based screening, and risk-stratified screening in the hypothetical cohort. Age-based screening consisted of screening every 4 years with prostate-specific antigen between the ages of 55 and 69 years. Risk-stratified screening used age and polygenic risk profiles.

**MAIN OUTCOMES AND MEASURES** The benefit-harm profile(deaths from prostate cancer, quality-adjusted life-years, overdiagnosis, and biopsies) and cost-effectiveness (net monetary benefit, analyzed from a health care system perspective) were analyzed. Both age-based and risk-stratified screening were evaluated using a biopsy-first and an MRI-first diagnostic pathway. Results were derived from probabilistic analyses and were discounted at 3.5% per annum.

## Getting started
This repository contains the code and underlying data used for this project.

### Prerequisites
To run the models:

Python 3.7+  
numpy
pandas
os
pathlib
scipy

### To run the code
Run the files main.py from your terminal or from a jupyter notebook having set the PATH variable to the base path in which your folder is stored in both main.py and utils/risk_distribution.py.  For example, if the folder is on your Desktop as ~/Desktop/project_folder/code... then set PATH to '~/Desktop/project_folder/'.  Follow the same process for sensitivity analyses using main_sensitivity_analyses.py

This will save the raw results in to a folder data/model_outputs.