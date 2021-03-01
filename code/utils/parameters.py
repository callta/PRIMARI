import pandas as pd
import numpy as np
from utils.functions import (psa_function, gamma, gamma_specified, normal,
                             lognormal, beta)


class Params():

    def __init__(self, PATH, sims):
        self.PATH = PATH
        self.sims = sims
        self.core_params = self._core_params()

    def _core_params(self):

        sims = self.sims

        """Core parameters which vary in sensitivity analyses.
        These parameters are generated automatically when the Params class is
        run so that they are available when the rest of the parameters
        (via the function gen_params) are generated."""

        np.random.seed(1234)

        # Detection ratio clinically significant cancers
        cs_mri = lognormal(0.0198, 0.0091, sims)
        cs_mri[cs_mri < 1] = 1
        self.cs_mri = cs_mri

        # Detection ratio clinically insignificant cancers
        self.cis_mri = lognormal(-0.0856, 0.0122, sims)

        # Uptake & Compliance
        self.uptake_prs = 1  # uptake of risk-stratification
        self.uptake_psa = 1  # uptake of PSA screening
        self.uptake_mri = 1  # uptake of MP-MRI
        self.compliance = 1  # compliance with recommendation to have or to forego PSA screening depending on risk.

        self.cost_mri = gamma(33.9, 11.2, sims)
        self.cost_prs = gamma(33.9, 0.7, sims)


    def gen_params(self):

        np.random.seed(1234)
        sims = self.sims
        PATH = self.PATH

        # Discount factor (3.5% as per NICE)
        years_in_model = np.arange(0, 45)
        self.discount_factor = 1.035 ** - years_in_model

        # Population size by age
        self.pop = pd.read_csv(f'{PATH}data/processed/population.csv', index_col=0, header=None)


        # COSTS
        cost_psa = gamma(33.9, 0.4, sims)
        self.cost_psa = np.tile(cost_psa, (45, 1)).T  # Extend cost_psa to be a matrix of length 45 x sims
        cost_biopsy = gamma(33.9, 17.1, sims)
        self.cost_biopsy = np.tile(cost_biopsy, (45, 1)).T
        self.cost_assessment = gamma(33.9, 16.1, sims)
        self.cost_assessment_nomri = gamma(33.9, 27.3, sims)
        cost_as = gamma(33.9, 149.1, sims)
        cost_rp = gamma(33.9, 289.5, sims)
        cost_rt = gamma(33.9, 190.7, sims)
        cost_brachytherapy = gamma(33.9, 54.1, sims)
        cost_adt = gamma(33.9, 19.8, sims)
        cost_chemo = gamma(33.9, 263, sims)

        cost_rt_chemo = cost_rt + cost_chemo
        cost_rp_rt = cost_rp + cost_rt
        cost_rp_chemo = cost_rp + cost_chemo
        cost_rp_rt_chemo = cost_rp + cost_rt + cost_chemo

        self.tx_costs = np.stack((cost_chemo, cost_rp,
                                  cost_rt, cost_rt_chemo,
                                  cost_rp_chemo, cost_rp_rt,
                                  cost_rp_rt_chemo, cost_as,
                                  cost_adt, cost_brachytherapy), axis=-1)

        # Incident costs / treatment dataframe
        self.tx = pd.read_csv(f'{PATH}data/processed/treatments.csv')

        self.cost_pca_death = gamma(1.8, 4625.9, sims)

        # Relative cost increase if clinically detected, source: Pharoah et al. 2013
        self.relative_cost_clinically_detected = normal(1.1, 0.04, sims)


        # UTILITIES - aged 45-90
        utility_background = np.array([
            0.8639, 0.85978064, 0.85568092, 0.85160074, 0.84754003,
            0.84349867, 0.83947659, 0.83547369, 0.83148987, 0.82752505,
            0.82357913, 0.81965203, 0.81574366, 0.81185392, 0.80798273,
            0.80413, 0.80029564, 0.79647957, 0.79268169, 0.78890191,
            0.78514017, 0.78139636, 0.7776704, 0.77396221, 0.7702717,
            0.76659879, 0.76294339, 0.75930542, 0.7556848, 0.75208144,
            0.74849527, 0.74492619, 0.74137413, 0.73783901, 0.73432075,
            0.73081927, 0.72733448, 0.7238663, 0.72041467, 0.71697949,
            0.71356069, 0.7101582, 0.70677193, 0.7034018, 0.70004775
        ])

        self.utility_background = gamma_specified((utility_background-0.03), 0.167, 4, 0.06, sims)

        utility_pca = np.array([
            0.803427, 0.79959599, 0.79578325, 0.79198869, 0.78821223,
            0.78445377, 0.78071323, 0.77699053, 0.77328558, 0.7695983,
            0.7659286, 0.76227639, 0.7586416, 0.75502415, 0.75142394,
            0.7478409, 0.74427495, 0.740726, 0.73719397, 0.73367878,
            0.73018036, 0.72669861, 0.72323347, 0.71978485, 0.71635268,
            0.71293687, 0.70953735, 0.70615404, 0.70278686, 0.69943574,
            0.6961006, 0.69278136, 0.68947794, 0.68619028, 0.6829183,
            0.67966192, 0.67642106, 0.67319566, 0.66998564, 0.66679093,
            0.66361144, 0.66044712, 0.65729789, 0.65416368, 0.65104441
        ])

        self.utility_pca = gamma_specified((utility_pca-0.05), 0.2, 5, 0.05, sims)


        # STAGE AT DIAGNOSIS
        # Proportion diagnosis at a localised stage aged 45 through 90 without screening
        stage_local_ns = np.array([
           0.532, 0.5272, 0.5224, 0.5176, 0.5128, 0.508, 0.5032, 0.4984,
           0.4936, 0.4888, 0.484, 0.4792, 0.4744, 0.4696, 0.4648, 0.46,
           0.4552, 0.4504, 0.4456, 0.4408, 0.436, 0.4312, 0.4264, 0.4216,
           0.4168, 0.412, 0.4072, 0.4024, 0.3976, 0.3928, 0.388, 0.3832,
           0.3784, 0.3736, 0.3688, 0.364, 0.3592, 0.3544, 0.3496, 0.3448,
           0.34, 0.3352, 0.3304, 0.3256, 0.3208
        ])

        self.stage_local_ns_psa = psa_function(stage_local_ns, sims)

        # Proportion diagnosis at an advanced stage aged 45 through 90 without screening
        stage_adv_ns = np.array([
           0.468, 0.4728, 0.4776, 0.4824, 0.4872, 0.492, 0.4968, 0.5016,
           0.5064, 0.5112, 0.516, 0.5208, 0.5256, 0.5304, 0.5352, 0.54,
           0.5448, 0.5496, 0.5544, 0.5592, 0.564, 0.5688, 0.5736, 0.5784,
           0.5832, 0.588, 0.5928, 0.5976, 0.6024, 0.6072, 0.612, 0.6168,
           0.6216, 0.6264, 0.6312, 0.636, 0.6408, 0.6456, 0.6504, 0.6552,
           0.66, 0.6648, 0.6696, 0.6744, 0.6792
        ])

        self.stage_adv_ns_psa = psa_function(stage_adv_ns, sims)

        # Stage at diagnosis (MRI-first)
        self.ns_localised_stage = self.stage_local_ns_psa * self.cis_mri
        self.ns_advanced_stage = 1-self.ns_localised_stage

        # Proportion diagnosis at a localised stage aged 45 through 90 with screening
        stage_local_screened = np.array([
           0.816, 0.8096, 0.8032, 0.7968, 0.7904, 0.784, 0.7776, 0.7712,
           0.7648, 0.7584, 0.752, 0.7456, 0.7392, 0.7328, 0.7264, 0.72,
           0.7136, 0.7072, 0.7008, 0.6944, 0.688, 0.6816, 0.6752, 0.6688,
           0.6624, 0.656, 0.6496, 0.6432, 0.6368, 0.6304, 0.624, 0.6176,
           0.6112, 0.6048, 0.5984, 0.592, 0.5856, 0.5792, 0.5728, 0.5664,
           0.56, 0.5536, 0.5472, 0.5408, 0.5344
        ])

        self.stage_local_screened_psa = psa_function(stage_local_screened, sims)

        # Proportion diagnosis at an advanced stage aged 45 through 90 with screening
        stage_advanced_screened = np.array([
            0.184, 0.1904, 0.1968, 0.2032, 0.2096, 0.216, 0.2224, 0.2288,
            0.2352, 0.2416, 0.248, 0.2544, 0.2608, 0.2672, 0.2736, 0.28,
            0.2864, 0.2928, 0.2992, 0.3056, 0.312, 0.3184, 0.3248, 0.3312,
            0.3376, 0.344, 0.3504, 0.3568, 0.3632, 0.3696, 0.376, 0.3824,
            0.3888, 0.3952, 0.4016, 0.408, 0.4144, 0.4208, 0.4272, 0.4336,
            0.44, 0.4464, 0.4528, 0.4592, 0.4656
        ])

        self.stage_adv_screened_psa = psa_function(stage_advanced_screened, sims)

        # Stage at diagnosis, adjusting for MRI first diagnostic pathway
        self.sc_localised_stage = self.stage_local_screened_psa * self.cis_mri
        self.sc_advanced_stage = 1-self.sc_localised_stage


        # Baseline incidence, prostate cancer mortality, death from other causes
        predicted_values = pd.read_csv(f'{PATH}data/processed/predicted_values.csv')

        # INCIDENCE
        self.pca_incidence = psa_function(predicted_values.pca_incidence.values, sims)
        self.adjusted_incidence = ((self.pca_incidence * self.stage_local_ns_psa * self.cis_mri)
                                   + (self.pca_incidence * self.stage_adv_ns_psa * self.cs_mri))


        # DEATH FROM PROSTATE CANCER
        self.pca_death_baseline = psa_function(predicted_values.pca_mortality.values, sims)

        # Adjusted baseline mortality
        # Assuming mortality is 4-fold greater in advanced vs local cancers
        # Assuming 50% of those detected at CS see a benefit
        mortality_local = (self.pca_death_baseline
                           / (self.stage_local_ns_psa + (self.stage_adv_ns_psa*4)))
        mortality_advanced = mortality_local * 4
        mri_impact_cs = ((abs(1-self.cs_mri) * self.stage_local_ns_psa * mortality_local)
                         + (abs(1-self.cs_mri) * (self.stage_adv_ns_psa/2) * mortality_advanced))

        self.adjusted_mortality = (2-self.cs_mri) * (self.pca_death_baseline+mri_impact_cs)


        # DEATH FROM OTHER CAUSES
        self.death_other_causes = psa_function(predicted_values.death_other_causes.values, sims)


        # The relative increase in cancers detected if screened
        p_increase_df = pd.read_csv(f'{PATH}data/processed/p_increase_df.csv', index_col='age')

        [RR_INCIDENCE_SC_55, RR_INCIDENCE_SC_56,
         RR_INCIDENCE_SC_57, RR_INCIDENCE_SC_58,
         RR_INCIDENCE_SC_59, RR_INCIDENCE_SC_60,
         RR_INCIDENCE_SC_61, RR_INCIDENCE_SC_62,
         RR_INCIDENCE_SC_63, RR_INCIDENCE_SC_64,
         RR_INCIDENCE_SC_65, RR_INCIDENCE_SC_66,
         RR_INCIDENCE_SC_67, RR_INCIDENCE_SC_68,
         RR_INCIDENCE_SC_69] = [np.random.lognormal(p_increase_df.loc[i, '1.23_log'],
                                                    p_increase_df.loc[i, 'se'],
                                                    sims)
                                for i in np.arange(55, 70, 1)]

        rr_incidence = np.vstack((np.array([np.repeat(1, sims)]*10),
                                  RR_INCIDENCE_SC_55, RR_INCIDENCE_SC_56, RR_INCIDENCE_SC_57,
                                  RR_INCIDENCE_SC_58, RR_INCIDENCE_SC_59, RR_INCIDENCE_SC_60,
                                  RR_INCIDENCE_SC_61, RR_INCIDENCE_SC_62, RR_INCIDENCE_SC_63,
                                  RR_INCIDENCE_SC_64, RR_INCIDENCE_SC_65, RR_INCIDENCE_SC_66,
                                  RR_INCIDENCE_SC_67, RR_INCIDENCE_SC_68, RR_INCIDENCE_SC_69))

        rr_incidence[rr_incidence < 1] = 1.03  # truncate
        self.rr_incidence = rr_incidence

        rr_incidence_adjusted = rr_incidence.copy()
        rr_incidence_adjusted[10:, :] = ((rr_incidence_adjusted[10:, :].T
                                          * self.stage_local_screened_psa[:, 10:25]
                                          * self.cis_mri[:, 10:25])

                                         + (rr_incidence_adjusted[10:, :].T
                                            * self.stage_adv_screened_psa[:, 10:25]
                                            * self.cs_mri[:, 10:25])).T

        self.rr_incidence_adjusted = rr_incidence_adjusted

        # Drop in incidence in the year after screening stops
        self.post_sc_incidence_drop = 0.9

        # Relative risk of death in screened cohort
        self.rr_death_screening = lognormal(-0.2231, 0.0552, sims)

        # Number of biopsies per cancer detected
        # Proportion having biopsy (screened arms)
        self.p_biopsy_sc = normal(0.24, 0.05, sims)

        # Proportion having biopsy (non-screened arms)
        # Ahmed et al. 2017, Table S6 (doi: 10.1016/S0140-6736(16)32401-1)
        # See appendix_workings_primari for further details
        self.p_biopsy_ns = normal(0.6, 0.1, sims)

        # PSA tests
        self.n_psa_tests = normal(1.2, 0.05, sims)

        # Proportion of PSA tests ≥ 3ng/ml age 45-90
        # Source: Leal et al. 2018 (doi: 10.1016/j.canep.2017.12.002) - proportion of screened population with PSA >/= 3ng/ml
        p_mri = np.array([
            0.1, 0.1, 0.1, 0.1, 0.00113, 0.01155, 0.02197,
            0.03239, 0.04281, 0.05323, 0.06365, 0.07407, 0.08449, 0.09491,
            0.10533, 0.11575, 0.12617, 0.13659, 0.14701, 0.15743, 0.16785,
            0.17827, 0.18869, 0.19911, 0.20953, 0.21995, 0.23037, 0.24079,
            0.25121, 0.26163, 0.27205, 0.28247, 0.29289, 0.30331, 0.31373,
            0.32415, 0.33457, 0.34499, 0.35541, 0.36583, 0.37625, 0.38667,
            0.39709, 0.40751, 0.41793
        ])

        self.p_mri = psa_function(p_mri, sims)

        # Proportion of newly diagnosed having an MRI in the biopsy-first group
        # 62% in the 2019 NPCA audit which covers 2017-2018 (pre-2019 guidelines)
        # 0.04 SD meaning 2.5th and 97.5th centiles 0.54 and 0.70
        self.mri_biopsy_first = normal(0.62, 0.04, sims)

        # Biopsies per MRI with an MRI-first approach to diagnosis (33% fewer biopsies)
        # 33% (95% CI: 23% - 45%) reduction - Elwenspoek et al. 2019 / Drost et al. 2019
        self.n_biopsies_primari = lognormal(-0.4005, 0.0632, sims)

        # Misclassification
        misclassified = beta(0.0276, 0.0036, sims)
        self.misclassified = np.tile(misclassified, (45, 1)).T  # extend to be sims,45

        # Estimates of mean sojourn time used to estimate when misclassified cancers would become clinical
        # Ranges from 11.3 to 12.6 years - Pashayan et al. 2009
        # Central estimate is 12 when rounded (11.3+(12.6-11.3)/2) which is needed for indexing
        self.mst = 12

        # Proportion of cancers at risk of overdiagnosis adjusted for reduction in CIS
        # Proportion overdiagnosed at ages 45 through 69
        overdiagnosis = np.array([
            0.0055, 0.0185, 0.0315, 0.0445, 0.0575, 0.0705, 0.0835, 0.0965,
            0.1095, 0.1225, 0.1355, 0.1485, 0.1615, 0.1745, 0.1875, 0.2005,
            0.2135, 0.2265, 0.2395, 0.2525, 0.2655, 0.2785, 0.2915, 0.3045,
            0.3175
        ])

        p_overdiagnosis = beta(overdiagnosis, 0.001, sims)
        additional_years = psa_function(np.repeat(0, 20), sims)
        p_overdiagnosis = np.concatenate((p_overdiagnosis, additional_years.T))
        self.p_overdiagnosis_psa = p_overdiagnosis * self.cis_mri.T
        self.p_overdiagnosis_psa[:10, :] = 0
        self.p_overdiagnosis_psa_nomri = p_overdiagnosis
        self.p_overdiagnosis_psa_nomri[:10, :] = 0
