import pandas as pd
from utils.parameters import Params
from models.no_screening import NoScreening
from models.no_screening_noMRI import NoScreeningNoMRI
from models.age_screening import AgeScreening
from models.age_screening_noMRI import AgeScreeningNoMRI
from models.prs_screening import PrsScreening
from models.prs_screening_noMRI import PrsScreeningNoMRI

PATH = '...'
params = Params(PATH=PATH, sims=10000)
params.gen_params()

reference_absolute_risk = ['2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0',
                           '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5',
                           '9.0', '9.5', '10.0']

# Run through each AR threshold in turn:
for reference_value in reference_absolute_risk:

    a_risk = pd.read_csv(f'{PATH}data/processed/a_risk/a_risk_{reference_value}.csv').set_index('age')
    pathname = f'{PATH}data/model_outputs/{reference_value}/'

    NoScreening(params).write_to_file(
                    PATH=pathname,
                    name='no_screening',
                    reference_value=reference_value)

    NoScreeningNoMRI(params).write_to_file(
                        PATH=pathname,
                        name='no_screening_noMRI',
                        reference_value=reference_value)

    AgeScreening(params).write_to_file(
                    PATH=pathname,
                    name='age_screening',
                    reference_value=reference_value)

    AgeScreeningNoMRI(params).write_to_file(
                        PATH=pathname,
                        name='age_screening_noMRI',
                        reference_value=reference_value)

    PrsScreening(params, a_risk).write_to_file(
                    PATH=pathname,
                    name='prs_screening',
                    reference_value=reference_value)

    PrsScreeningNoMRI(params, a_risk).write_to_file(
                        PATH=pathname,
                        name='prs_screening_noMRI',
                        reference_value=reference_value)
