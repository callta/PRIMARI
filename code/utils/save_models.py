import pandas as pd
import os


class Save():

    def write_to_file(self, PATH, name, reference_value):

        if os.path.isdir(PATH):
            pass
        else:
            os.makedirs(PATH)

        # write the dataframes to an excel file - one sheet for each cohort
        def save_excel(list_dataframes, name):
            writer = pd.ExcelWriter(name+'.xlsx', engine='xlsxwriter')
            for i, df in enumerate(list_dataframes):
                df.to_excel(writer, 'cohort_%s' % (i+55), index=False)
            writer.save()

        save_excel(self.cohort_list, PATH+name+'_cohorts')

        # Save the collated outcome dataframes
        outcomes = pd.concat(self.outcomes_list)
        outcomes.to_excel(PATH+'outcomes'+'_'+name+'_'+reference_value+'.xlsx',
                          engine='xlsxwriter', sheet_name=name, index=False)

        # Set path to store outputs of models
        path = (PATH+"simulation dataframes/")
        os.makedirs(path, exist_ok=True)

        # Raw 10,000 simulations
        varnames = ['qalys', 'lyrs', 'costs', 'pca_deaths',
                    'pca_cases', 'cost_mri', 'cost_biopsy', 'n_mri',
                    'n_biopsies', 'n_psa_tests', 'overdiagnosis']

        for varname in varnames:
            pd.DataFrame(self.simulations[varname]).to_csv(
                path+name+'_'+varname+'_'+reference_value+'.csv', index=False)
