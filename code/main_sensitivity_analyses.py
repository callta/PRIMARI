import pandas as pd
from utils.functions import psa_function, lognormal
from utils.parameters import Params
from models.no_screening import NoScreening
from models.no_screening_noMRI import NoScreeningNoMRI
from models.age_screening import AgeScreening
from models.age_screening_noMRI import AgeScreeningNoMRI
from models.prs_screening import PrsScreening
from models.prs_screening_noMRI import PrsScreeningNoMRI

sims = 10000
PATH = '...'
params = Params(PATH=PATH, sims=sims)
params.gen_params()

reference_absolute_risk = ['2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0',
                           '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5',
                           '9.0', '9.5', '10.0']

def run_analyses(params, sensitivity_analysis, od_by_risk:bool=False):

    for reference_value in reference_absolute_risk:

        pathname = f'{PATH}data/model_outputs/sensitivity_analyses/{sensitivity_analysis}{reference_value}/'
        a_risk = pd.read_csv(f'{PATH}data/processed/a_risk/a_risk_{reference_value}.csv').set_index('age')

        NoScreening(params).write_to_file(PATH=pathname, name='no_screening',
                                          reference_value=reference_value)

        NoScreeningNoMRI(params).write_to_file(PATH=pathname, name='no_screening_noMRI',
                                               reference_value=reference_value)

        AgeScreening(params).write_to_file(PATH=pathname, name='age_screening',
                                           reference_value=reference_value)

        AgeScreeningNoMRI(params).write_to_file(PATH=pathname, name='age_screening_noMRI',
                                                reference_value=reference_value)

        PrsScreening(params, a_risk, od_by_risk).write_to_file(PATH=pathname, name='prs_screening',
                                                   reference_value=reference_value)

        PrsScreeningNoMRI(params, a_risk, od_by_risk).write_to_file(PATH=pathname, name='prs_screening_noMRI',
                                                                    reference_value=reference_value)


# PRS cost
# £50 per PRS
prs = Params(PATH=PATH, sims=sims)
prs.gen_params()
prs.cost_prs = psa_function(50, sims)
run_analyses(prs, 'prs_cost/50/')

# £100 per PRS
prs.cost_prs = psa_function(100, sims)
run_analyses(prs, 'prs_cost/100/')

# MRI cost
# £100 per MRI
mri = Params(PATH=PATH, sims=sims)
mri.gen_params()
mri.cost_mri = psa_function(100, sims)
run_analyses(mri, 'mri_cost/100/')

# £200 per MRI
mri.cost_mri = psa_function(200, sims)
run_analyses(mri, 'mri_cost/200/')

# Using assumptions from the PRECISION study regarding impact of MRI prior to biopsy on clinically significant & insignificant cancers detected
precision = Params(PATH=PATH, sims=sims)
precision.cs_mri = lognormal(0.1133, 0.0373, sims)
precision.cs_mri[precision.cs_mri < 1] = 1
precision.cis_mri = lognormal(-0.1393, 0.03596, sims)
precision.gen_params()
run_analyses(precision, 'PRECISION/')

# Overdiagnosis varying with polygenic risk
od = Params(PATH=PATH, sims=sims)
od.gen_params()
run_analyses(od, 'overdiagnosis/', od_by_risk=True)

# Varying uptake with screening
uptake_prs = Params(PATH=PATH, sims=sims)
uptake_prs.uptake_prs = 0.75
uptake_prs.gen_params()
run_analyses(uptake_prs, 'uptake_prs/')

uptake_psa = Params(PATH=PATH, sims=sims)
uptake_psa.uptake_psa = 0.75
uptake_psa.gen_params()
run_analyses(uptake_psa, 'uptake_psa/')
