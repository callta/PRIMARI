import pandas as pd
import numpy as np
from collections import defaultdict
from utils.save_models import Save
from utils.functions import mean, total


class AgeScreeningNoMRI(Save):

    """Build cohort of age-based screening using biopsy-first approach
    to diagnosis, i.e. pre-2019 NICE guidelines."""

    def __init__(self, params):
        self.cohort_list = []
        self.outcomes_list = []
        self.simulations = defaultdict(list)
        self.run_model = self._run_model(params)

    def _run_model(self, params):

        # Loop through age cohorts
        for year in np.arange(55, 70):

            if year < 55:
                # This exists if were to extend to cohorts from age 45
                # PCa incidence non-screened
                pca_incidence_age = params.pca_incidence[:, year-45:]
                pca_mortality_age = params.pca_death_baseline[:, year-45:]

            if year > 54:
                # PCa incidence
                pca_incidence_age = params.pca_incidence.copy()
                pca_incidence_age[:,10:25] = (pca_incidence_age[:,10:25].T * params.rr_incidence[year-45,:]).T
                pca_incidence_age[:,25:35] = pca_incidence_age[:,25:35] * np.linspace(params.post_sc_incidence_drop,1,10)
                pca_incidence_age = pca_incidence_age[:, year-45:]

                # Death from PCa
                pca_mortality_age = params.pca_death_baseline.copy()
                pca_mortality_age[:,10:15] = pca_mortality_age[:,10:15] * np.linspace(1,0.8,5)
                pca_mortality_age[:,15:] = pca_mortality_age[:,15:] * params.rr_death_screening[:,15:]
                pca_mortality_age = pca_mortality_age[:, year-45:]

                # Proportion of cancers detected by screening at a localised / advanced stage
                stage_screened_adv_nomri = params.stage_adv_screened_psa[:, year-45:]
                stage_screened_local_nomri = params.stage_local_screened_psa[:, year-45:]

            # Parameters for the non-screened
            incidence = params.pca_incidence[:, year-45:]
            pca_mortality = params.pca_death_baseline[:, year-45:]
            mortality_other_causes = params.death_other_causes[:, year-45:]

            stage_ns_local_nomri = params.stage_local_ns_psa[:, year-45:]
            stage_ns_adv_nomri = params.stage_adv_ns_psa[:, year-45:]

            tx_costs_local = params.tx_costs * params.tx.localised.values
            tx_costs_adv = params.tx_costs * params.tx.advanced.values

            #######################
            # Year 1 in the model #
            #######################

            age = np.arange(year,90)
            length_df = len(age)
            length_screen = len(np.arange(year,70)) # number of screening years depending on age cohort starting

            # Cohorts, numbers healthy, and incident cases
            cohort_sc = np.array([np.repeat(params.pop.loc[year, :], length_df)] * params.sims) * params.uptake_psa
            cohort_ns = np.array([np.repeat(params.pop.loc[year, :], length_df)] * params.sims) * (1-params.uptake_psa)

            pca_alive_sc = np.array([np.zeros(length_df)] * params.sims)
            pca_alive_ns = np.array([np.zeros(length_df)] * params.sims)

            healthy_sc = cohort_sc - pca_alive_sc
            healthy_ns = cohort_ns - pca_alive_ns

            pca_incidence_sc = healthy_sc * pca_incidence_age # Total incidence in screened arm

            if year > 54:
                pca_incidence_screened = pca_incidence_sc.copy()
                pca_incidence_post_screening = np.array([np.zeros(length_df)] * params.sims) # Post-screening cancers - 0 until model reaches age 70.

            elif year < 55:
                pca_incidence_screened = np.array([np.zeros(length_df)] * params.sims)
                pca_incidence_post_screening = np.array([np.zeros(length_df)] * params.sims) # Post-screening cancers 0 as no screening (needed for later code to run smoothly)

            pca_incidence_ns = healthy_ns * incidence # Incidence in non-screened

            # Deaths
            pca_death_sc = ((pca_alive_sc * pca_mortality_age)
                            + (healthy_sc * pca_mortality_age))

            pca_death_ns = ((pca_alive_ns * pca_mortality)
                            + (healthy_ns * pca_mortality))

            pca_death_other_sc = ((pca_incidence_sc
                                   + pca_alive_sc
                                   - pca_death_sc)
                                  * mortality_other_causes)

            pca_death_other_ns = ((pca_incidence_ns
                                   + pca_alive_ns
                                   - pca_death_ns)
                                  * mortality_other_causes)

            healthy_death_other_sc = ((healthy_sc - pca_incidence_sc)
                                      * mortality_other_causes)

            healthy_death_other_ns = ((healthy_ns - pca_incidence_ns)
                                      * mortality_other_causes)

            t_death_sc = (pca_death_sc
                          + pca_death_other_sc
                          + healthy_death_other_sc) # Total deaths screened arm

            t_death_ns = (pca_death_ns
                          + pca_death_other_ns
                          + healthy_death_other_ns) # Total deaths non-screened arm

            t_death = t_death_sc + t_death_ns # Total deaths

            # Prevalent cases & life-years
            pca_prevalence_sc = (pca_incidence_sc
                                 - pca_death_sc
                                 - pca_death_other_sc)

            pca_prevalence_ns = (pca_incidence_ns
                                 - pca_death_ns
                                 - pca_death_other_ns)

            lyrs_pca_sc_nodiscount = pca_prevalence_sc * 0.5
            lyrs_pca_ns_nodiscount = pca_prevalence_ns * 0.5

            # Costs
            if year > 54:
                costs_tx_screened = np.array([np.zeros(length_df)] * params.sims)
                costs_tx_post_screening = np.array([np.zeros(length_df)] * params.sims)

                costs_tx_screened[:, 0] = ((pca_incidence_screened[:, 0]
                                           * stage_screened_local_nomri[:, 0].T
                                           * tx_costs_local.T).sum(axis=0)

                                          + (pca_incidence_screened[:, 0]
                                            * stage_screened_adv_nomri[:, 0].T
                                            * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

                costs_tx_post_screening[:, 0] = ((pca_incidence_post_screening[:, 0]
                                                 * stage_ns_local_nomri[:, 0].T
                                                 * tx_costs_local.T).sum(axis=0)

                                                + (pca_incidence_post_screening[:, 0]
                                                   * stage_ns_adv_nomri[:, 0].T
                                                   * tx_costs_adv.T).sum(axis=0)

                                               * params.relative_cost_clinically_detected[:, 0])

                costs_tx_sc = np.array([np.zeros(length_df)] * params.sims)
                costs_tx_sc[:, 0] = (costs_tx_screened[:, 0] + costs_tx_post_screening[:, 0]) # total cost in screened arms

            elif year < 55:
                costs_tx_sc = np.array([np.zeros(length_df)] * params.sims)
                costs_tx_sc[:, 0] =  ((pca_incidence_sc[:, 0]
                                     * stage_ns_local_nomri[:, 0].T
                                     * tx_costs_local.T).sum(axis=0)

                                    + (pca_incidence_sc[:, 0]
                                       * stage_ns_adv_nomri[:, 0].T
                                       * tx_costs_adv.T).sum(axis=0)

                                    * params.relative_cost_clinically_detected[:, 0])

            costs_tx_ns = np.array([np.zeros(length_df)] * params.sims)
            costs_tx_ns[:, 0] =  ((pca_incidence_ns[:, 0]
                                 * stage_ns_local_nomri[:, 0].T
                                 * tx_costs_local.T).sum(axis=0)

                                + (pca_incidence_ns[:, 0]
                                   * stage_ns_adv_nomri[:, 0].T
                                   * tx_costs_adv.T).sum(axis=0)

                                * params.relative_cost_clinically_detected[:, 0])


            ##################
            # Year 2 onwards #
            ##################
            total_cycles = length_df
            for i in range(1, total_cycles):

               # Cohorts, numbers healthy, incident & prevalent cases
                cohort_sc[:, i] = cohort_sc[:, i-1] - t_death_sc[:, i-1]
                cohort_ns[:, i] = cohort_ns[:, i-1] - t_death_ns[:, i-1]

                pca_alive_sc[:, i] = (pca_alive_sc[:, i-1]
                                     + pca_incidence_sc[:, i-1]
                                     - pca_death_sc[:, i-1]
                                     - pca_death_other_sc[:, i-1])

                pca_alive_ns[:, i] = (pca_alive_ns[:, i-1]
                                     + pca_incidence_ns[:, i-1]
                                     - pca_death_ns[:, i-1]
                                     - pca_death_other_ns[:, i-1])

                healthy_sc[:, i] = cohort_sc[:, i] - pca_alive_sc[:, i]
                healthy_ns[:, i] = cohort_ns[:, i] - pca_alive_ns[:, i]

                pca_incidence_sc[:, i] = healthy_sc[:, i] * pca_incidence_age[:, i]

                if year > 54:
                    if i < length_screen:
                        pca_incidence_screened[:, i] = pca_incidence_sc[:, i].copy()
                        pca_incidence_post_screening[:, i] = 0

                    else:
                        pca_incidence_screened[:, i] = 0 # Screen-detected cancers
                        pca_incidence_post_screening[:, i] = pca_incidence_sc[:, i].copy()

                elif year < 55:
                    pca_incidence_screened[:, i] = 0 # Screen-detected cancers
                    pca_incidence_post_screening[:, i] = 0 # post-screening cancers 0 as no screening (needed for later code to run smoothly)

                pca_incidence_ns[:, i] = healthy_ns[:, i] * incidence[:, i]

                # Deaths
                pca_death_sc[:, i] = ((pca_alive_sc[:, i] * pca_mortality_age[:, i])
                                     + (healthy_sc[:, i] * pca_mortality_age[:, i]))

                pca_death_ns[:, i] = ((pca_alive_ns[:, i] * pca_mortality[:, i])
                                     + (healthy_ns[:, i] * pca_mortality[:, i]))

                pca_death_other_sc[:, i] = ((pca_incidence_sc[:, i]
                                            + pca_alive_sc[:, i]
                                            - pca_death_sc[:, i])
                                           * mortality_other_causes[:, i])

                pca_death_other_ns[:, i] = ((pca_incidence_ns[:, i]
                                            + pca_alive_ns[:, i]
                                            - pca_death_ns[:, i])
                                           * mortality_other_causes[:, i])

                healthy_death_other_sc[:, i] = ((healthy_sc[:, i] - pca_incidence_sc[:, i])
                                               * mortality_other_causes[:, i])

                healthy_death_other_ns[:, i] = ((healthy_ns[:, i] - pca_incidence_ns[:, i])
                                               * mortality_other_causes[:, i])

                t_death_sc[:, i] = (pca_death_sc[:, i]
                                   + pca_death_other_sc[:, i]
                                   + healthy_death_other_sc[:, i])

                t_death_ns[:, i] = (pca_death_ns[:, i]
                                   + pca_death_other_ns[:, i]
                                   + healthy_death_other_ns[:, i])

                t_death[:, i] = t_death_sc[:, i] + t_death_ns[:, i]

                # Prevalent cases & life-years
                pca_prevalence_sc[:, i] = (pca_incidence_sc[:, i]
                                          + pca_alive_sc[:, i]
                                          - pca_death_sc[:, i]
                                          - pca_death_other_sc[:, i])

                pca_prevalence_ns[:, i] = (pca_incidence_ns [:, i]
                                          + pca_alive_ns[:, i]
                                          - pca_death_ns[:, i]
                                          - pca_death_other_ns[:, i])

                lyrs_pca_sc_nodiscount[:, i] = (pca_prevalence_sc[:, i-1] + pca_prevalence_sc[:, i]) * 0.5
                lyrs_pca_ns_nodiscount[:, i] = (pca_prevalence_ns[:, i-1] + pca_prevalence_ns[:, i]) * 0.5

                # Costs
                if year > 54:
                    costs_tx_screened[:, i] = ((pca_incidence_screened[:, i]
                                                * stage_screened_local_nomri[:, i].T
                                                * tx_costs_local.T).sum(axis=0)

                                              + (pca_incidence_screened[:, i]
                                                * stage_screened_adv_nomri[:, i].T
                                                * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

                    costs_tx_post_screening[:, i] = ((pca_incidence_post_screening[:, i]
                                                     * stage_ns_local_nomri[:, i].T
                                                     * tx_costs_local.T).sum(axis=0)

                                                    + (pca_incidence_post_screening[:, i]
                                                       * stage_ns_adv_nomri[:, i].T
                                                       * tx_costs_adv.T).sum(axis=0)

                                                   * params.relative_cost_clinically_detected[:, i])

                    costs_tx_sc[:, i] = costs_tx_screened[:, i] + costs_tx_post_screening[:, i] # total cost in screened arms

                elif year < 55:
                    costs_tx_sc[:, i] =  ((pca_incidence_sc[:, i]
                                         * stage_ns_local_nomri[:, i].T
                                         * tx_costs_local.T).sum(axis=0)

                                        + (pca_incidence_sc[:, i]
                                           * stage_ns_adv_nomri[:, i].T
                                           * tx_costs_adv.T).sum(axis=0)

                                        * params.relative_cost_clinically_detected[:, i])

                costs_tx_ns[:, i] =  ((pca_incidence_ns[:, i]
                                     * stage_ns_local_nomri[:, i].T
                                     * tx_costs_local.T).sum(axis=0)

                                    + (pca_incidence_ns[:, i]
                                       * stage_ns_adv_nomri[:, i].T
                                       * tx_costs_adv.T).sum(axis=0)

                                    * params.relative_cost_clinically_detected[:, i])

            ############
            # Outcomes #
            ############

            # Incident cases (total)
            cases_age = pca_incidence_sc + pca_incidence_ns

            # PCa alive
            pca_alive_age = pca_alive_sc + pca_alive_ns

            # Healthy
            healthy_age = healthy_sc + healthy_ns

            # Overdiagnosed cases
            overdiagnosis_age = pca_incidence_screened * params.p_overdiagnosis_psa_nomri.T[:, year-45:]

            # Deaths from other causes (screened arm)
            deaths_sc_other_age = pca_death_other_sc + healthy_death_other_sc

            # Deaths from other causes (non-screened arm)
            deaths_ns_other_age = pca_death_other_ns + healthy_death_other_ns

            # Deaths from other causes (total)
            deaths_other_age = deaths_sc_other_age + deaths_ns_other_age

            # Deaths from prosate cancer (total)
            ####################################
            deaths_pca_age = pca_death_sc + pca_death_ns

            ##############
            # Life-years #
            ##############

            # Healthy life-years (screened arm)
            lyrs_healthy_sc_nodiscount_age = healthy_sc - (0.5 * (healthy_death_other_sc+pca_incidence_sc))
            lyrs_healthy_sc_discount_age = lyrs_healthy_sc_nodiscount_age * params.discount_factor[:total_cycles]

            # Healthy life-years (non-screened arm)
            lyrs_healthy_ns_nodiscount_age = healthy_ns - (0.5 * (healthy_death_other_ns+pca_incidence_ns))
            lyrs_healthy_ns_discount_age = lyrs_healthy_ns_nodiscount_age * params.discount_factor[:total_cycles]

            # Total healthy life-years
            lyrs_healthy_nodiscount_age = lyrs_healthy_sc_nodiscount_age + lyrs_healthy_ns_nodiscount_age
            lyrs_healthy_discount_age = lyrs_healthy_nodiscount_age * params.discount_factor[:total_cycles]

            # Life-years with prostate cancer in screened arm
            lyrs_pca_sc_discount = lyrs_pca_sc_nodiscount * params.discount_factor[:total_cycles]

            # Life-years with prostate cancer in non-screened arm
            lyrs_pca_ns_discount = lyrs_pca_ns_nodiscount * params.discount_factor[:total_cycles]

            #  Life-years with prostate cancer in both arms
            lyrs_pca_nodiscount_age = lyrs_pca_sc_nodiscount + lyrs_pca_ns_nodiscount
            lyrs_pca_discount_age = lyrs_pca_sc_discount + lyrs_pca_ns_discount

            # Total life-years
            ##################
            lyrs_nodiscount_age = lyrs_healthy_nodiscount_age + lyrs_pca_nodiscount_age
            lyrs_discount_age = lyrs_healthy_discount_age + lyrs_pca_discount_age

            #########
            # QALYs #
            #########

            # QALYs (healthy life) - screened arm
            qalys_healthy_sc_nodiscount_age = lyrs_healthy_sc_nodiscount_age * params.utility_background[:, year-45:]
            qalys_healthy_sc_discount_age = lyrs_healthy_sc_discount_age * params.utility_background[:, year-45:]

            # QALYs (healthy life) - non-screened arm
            qalys_healthy_ns_nodiscount_age = lyrs_healthy_ns_nodiscount_age * params.utility_background[:, year-45:]
            qalys_healthy_ns_discount_age = lyrs_healthy_ns_discount_age * params.utility_background[:, year-45:]

            # Total QALYs (healthy life)
            qalys_healthy_nodiscount_age = lyrs_healthy_nodiscount_age * params.utility_background[:, year-45:]
            qalys_healthy_discount_age = lyrs_healthy_discount_age * params.utility_background[:, year-45:]

            # QALYS with prostate cancer - screened arm
            qalys_pca_sc_nodiscount_age = lyrs_pca_sc_nodiscount * params.utility_pca[:, year-45:]
            qalys_pca_sc_discount_age = lyrs_pca_sc_discount * params.utility_pca[:, year-45:]

            # QALYS with prostate cancer - non-screened arm
            qalys_pca_ns_nodiscount_age = lyrs_pca_ns_nodiscount * params.utility_pca[:, year-45:]
            qalys_pca_ns_discount_age = lyrs_pca_ns_discount * params.utility_pca[:, year-45:]

            # Total QALYS with prostate cancer
            qalys_pca_nodiscount_age = lyrs_pca_nodiscount_age * params.utility_pca[:, year-45:]
            qalys_pca_discount_age = lyrs_pca_discount_age * params.utility_pca[:, year-45:]

            # Total QALYs
            #############
            qalys_nodiscount_age = qalys_healthy_nodiscount_age + qalys_pca_nodiscount_age
            qalys_discount_age = qalys_healthy_discount_age + qalys_pca_discount_age

            #############
            # PSA tests #
            #############

            # Costs of PSA testing in non-screened arm
            n_psa_tests_ns_age = (pca_incidence_ns / params.p_biopsy_ns[:, year-45:]) * params.n_psa_tests[:, year-45:]
            cost_psa_testing_ns_nodiscount_age = (n_psa_tests_ns_age
                                                  * params.cost_psa[:, year-45:]
                                                  * params.relative_cost_clinically_detected[:, year-45:])

            cost_psa_testing_ns_discount_age = cost_psa_testing_ns_nodiscount_age * params.discount_factor[:total_cycles]

            # Costs of PSA testing in screened arm (PSA screening every four years)
            if year < 55:
                # Assuming all cancers are clinically detected as these cohorts are not eligible for screening (hence params.p_biopsy_ns)
                n_psa_tests_sc_age = (pca_incidence_sc / params.p_biopsy_ns[:, year-45:]) * params.n_psa_tests[:, year-45:]
                cost_psa_testing_sc_nodiscount_age = (n_psa_tests_sc_age
                                                      * params.cost_psa[:, year-45:]
                                                      * params.relative_cost_clinically_detected[:, year-45:])

            if year > 54:
                # Healthy people eligible for screening
                lyrs_healthy_screened_nodiscount_age = np.array([np.zeros(length_df)] * params.sims)
                lyrs_healthy_screened_nodiscount_age[:,:length_screen] = lyrs_healthy_sc_nodiscount_age[:,:length_screen].copy()
                lyrs_healthy_screened_nodiscount_age[:,length_screen:] = 0

                # Population-level PSA testing during screening phase
                n_psa_tests_screened_age = lyrs_healthy_screened_nodiscount_age * params.uptake_psa / 4
                cost_psa_testing_screened_age = n_psa_tests_screened_age * params.cost_psa[:, year-45:]

                # Assuming all cancers are clinically detected in the post-screening phase
                n_psa_tests_post_screening_age = (pca_incidence_post_screening / params.p_biopsy_ns[:, year-45:]) * params.n_psa_tests[:, year-45:]
                cost_psa_testing_post_screening_age = (n_psa_tests_post_screening_age
                                                       * params.cost_psa[:, year-45:]
                                                       * params.relative_cost_clinically_detected[:, year-45:])
                # Total PSA tests
                n_psa_tests_sc_age = n_psa_tests_screened_age + n_psa_tests_post_screening_age

                # Total PSA costs
                cost_psa_testing_sc_nodiscount_age = cost_psa_testing_screened_age + cost_psa_testing_post_screening_age

            cost_psa_testing_sc_discount_age = cost_psa_testing_sc_nodiscount_age * params.discount_factor[:total_cycles]

            # Total PSA testing
            ###################
            n_psa_tests_age = n_psa_tests_ns_age + n_psa_tests_sc_age
            cost_psa_testing_nodiscount_age = cost_psa_testing_ns_nodiscount_age + cost_psa_testing_sc_nodiscount_age
            cost_psa_testing_discount_age = cost_psa_testing_ns_discount_age + cost_psa_testing_sc_discount_age

            #######
            # MRI #
            #######

            if year < 55:
                n_mri_sc_age = pca_incidence_sc * params.mri_biopsy_first[:, year-45:]
                cost_mri_sc_nodiscount_age = (n_mri_sc_age
                                              * np.array([params.cost_mri]).T
                                              * params.relative_cost_clinically_detected[:, year-45:])

            if year > 54:
                n_mri_screened_age = pca_incidence_screened * params.mri_biopsy_first[:, year-45:]
                cost_mri_screened_nodiscount_age = n_mri_screened_age * np.array([params.cost_mri]).T

                n_mri_post_screening_age = pca_incidence_post_screening * params.mri_biopsy_first[:, year-45:]
                cost_mri_post_screening_nodiscount_age = (n_mri_post_screening_age
                                                          * np.array([params.cost_mri]).T
                                                          * params.relative_cost_clinically_detected[:, year-45:])

                # Total MRI
                n_mri_sc_age = n_mri_screened_age + n_mri_post_screening_age
                cost_mri_sc_nodiscount_age = cost_mri_screened_nodiscount_age + cost_mri_post_screening_nodiscount_age

            cost_mri_sc_discount_age = cost_mri_sc_nodiscount_age * params.discount_factor[:total_cycles]

            # Costs of mri - non-screened arm
            n_mri_ns_age = pca_incidence_ns * params.mri_biopsy_first[:, year-45:]
            cost_mri_ns_nodiscount_age = (n_mri_ns_age
                                          * np.array([params.cost_mri]).T
                                          * params.relative_cost_clinically_detected[:, year-45:])

            cost_mri_ns_discount_age = cost_mri_ns_nodiscount_age * params.discount_factor[:total_cycles]

            # Total costs of mri
            #######################
            n_mri_age = n_mri_sc_age + n_mri_ns_age
            cost_mri_nodiscount_age = cost_mri_sc_nodiscount_age + cost_mri_ns_nodiscount_age
            cost_mri_discount_age = cost_mri_sc_discount_age + cost_mri_ns_discount_age

            ##########
            # Biopsy #
            ##########

            if year < 55:
                n_biopsies_sc_age = pca_incidence_sc / params.p_biopsy_ns[:, year-45:]
                cost_biopsy_sc_nodiscount_age = (n_biopsies_sc_age * params.cost_biopsy[:, year-45:]) * params.relative_cost_clinically_detected[:, year-45:]

            if year > 54:
                # Screen-detected cancers
                n_biopsies_screened_age = pca_incidence_screened / params.p_biopsy_sc[:, year-45:]
                cost_biopsy_screened_nodiscount_age = n_biopsies_screened_age * params.cost_biopsy[:, year-45:]

                # Assuming all cancers are clinically detected in the post-screening phase
                n_biopsies_post_screening_age = pca_incidence_post_screening / params.p_biopsy_ns[:, year-45:]
                cost_biopsies_post_screening_nodiscount_age = (n_biopsies_post_screening_age * params.cost_biopsy[:, year-45:]) * params.relative_cost_clinically_detected[:, year-45:]

                # Total biopsies
                n_biopsies_sc_age = n_biopsies_screened_age + n_biopsies_post_screening_age
                cost_biopsy_sc_nodiscount_age = cost_biopsy_screened_nodiscount_age + cost_biopsies_post_screening_nodiscount_age

            cost_biopsy_sc_discount_age = cost_biopsy_sc_nodiscount_age * params.discount_factor[:total_cycles]

            # Costs of biopsy - non-screened arm
            n_biopsies_ns_age = pca_incidence_ns / params.p_biopsy_ns[:, year-45:]
            cost_biopsy_ns_nodiscount_age = (n_biopsies_ns_age * params.cost_biopsy[:, year-45:]) * params.relative_cost_clinically_detected[:, year-45:]
            cost_biopsy_ns_discount_age = cost_biopsy_ns_nodiscount_age * params.discount_factor[:total_cycles]

            # Total costs of biopsy
            #######################
            n_biopsies_age = n_biopsies_sc_age + n_biopsies_ns_age
            cost_biopsy_nodiscount_age = cost_biopsy_sc_nodiscount_age + cost_biopsy_ns_nodiscount_age
            cost_biopsy_discount_age = cost_biopsy_sc_discount_age + cost_biopsy_ns_discount_age

            # Cost of staging in the screened arm
            if year < 55:
                cost_staging_sc_nodiscount_age = (params.cost_assessment
                                                  * stage_ns_adv_nomri.T
                                                  * pca_incidence_sc.T
                                                  * params.relative_cost_clinically_detected[:, year-45:].T).T

            if year > 54:
                cost_staging_screened_nodiscount_age = (params.cost_assessment
                                                        * stage_screened_adv_nomri.T
                                                        * pca_incidence_screened.T).T

                cost_staging_post_screening_nodiscount_age = (params.cost_assessment
                                                              * stage_ns_adv_nomri.T
                                                              * pca_incidence_post_screening.T
                                                              * params.relative_cost_clinically_detected[:, year-45:].T).T

                cost_staging_sc_nodiscount_age = cost_staging_screened_nodiscount_age + cost_staging_post_screening_nodiscount_age

            cost_staging_sc_discount_age = cost_staging_sc_nodiscount_age * params.discount_factor[:total_cycles]

            # Cost of staging in the non-screened arm
            cost_staging_ns_nodiscount_age = (params.cost_assessment
                                              * stage_ns_adv_nomri.T
                                              * pca_incidence_ns.T
                                              * params.relative_cost_clinically_detected[:, year-45:].T).T

            cost_staging_ns_discount_age = cost_staging_ns_nodiscount_age * params.discount_factor[:total_cycles]

            # Total costs of staging
            ########################
            cost_staging_nodiscount_age = cost_staging_sc_nodiscount_age + cost_staging_ns_nodiscount_age
            cost_staging_discount_age = cost_staging_sc_discount_age + cost_staging_ns_discount_age

            # Cost of treatment in screened arm
            cost_tx_sc_nodiscount_age = costs_tx_sc * params.discount_factor[:total_cycles]

            # Cost of treatment in non-screened arm
            cost_tx_ns_nodiscount_age = costs_tx_ns * params.discount_factor[:total_cycles]

            # Total costs of treatment
            ##########################
            cost_tx_nodiscount_age = costs_tx_sc + costs_tx_ns
            cost_tx_discount_age = cost_tx_nodiscount_age * params.discount_factor[:total_cycles]

            # Costs of palliation and death in screened arm
            cost_eol_sc_nodiscount_age = (params.cost_pca_death * pca_death_sc.T).T
            cost_eol_sc_discount_age = cost_eol_sc_nodiscount_age * params.discount_factor[:total_cycles]

            # Costs of palliation and death in non-screened arm
            cost_eol_ns_nodiscount_age = (params.cost_pca_death * pca_death_ns.T).T
            cost_eol_ns_discount_age = cost_eol_ns_nodiscount_age * params.discount_factor[:total_cycles]

            # Total costs of palliation and death
            cost_eol_nodiscount_age = cost_eol_sc_nodiscount_age + cost_eol_ns_nodiscount_age
            cost_eol_discount_age = cost_eol_sc_discount_age + cost_eol_ns_discount_age

            # TOTAL COSTS AGE-BASED SCREENING
            #################################
            cost_nodiscount_age = (cost_psa_testing_nodiscount_age
                                   + cost_mri_nodiscount_age
                                   + cost_biopsy_nodiscount_age
                                   + cost_staging_nodiscount_age
                                   + cost_tx_nodiscount_age
                                   + cost_eol_nodiscount_age)

            cost_discount_age = (cost_psa_testing_discount_age
                                 + cost_mri_nodiscount_age
                                 + cost_biopsy_discount_age
                                 + cost_staging_discount_age
                                 + cost_tx_discount_age
                                 + cost_eol_discount_age)

            #######################################
            # Generate dataframes of the outcomes #
            #######################################

            # Generate a mean dataframe for each age cohort
            cohort = pd.DataFrame({
                'age': age,
                'pca_cases': mean(cases_age),
                'pca_cases_sc': mean(pca_incidence_sc),
                'pca_cases_ns': mean(pca_incidence_ns),
                'screen-detected cases': mean(pca_incidence_screened),
                'post-screening cases': mean(pca_incidence_post_screening),
                'overdiagnosis': mean(overdiagnosis_age),
                'deaths_other': mean(deaths_other_age),
                'deaths_other_sc': mean(deaths_sc_other_age),
                'deaths_other_ns': mean(deaths_ns_other_age),
                'deaths_pca': mean(deaths_pca_age),
                'deaths_pca_sc': mean(pca_death_sc),
                'deaths_pca_ns': mean(pca_death_ns),
                'pca_alive': mean(pca_alive_age),
                'pca_alive_sc': mean(pca_alive_sc),
                'pca_alive_ns': mean(pca_alive_ns),
                'healthy': mean(healthy_age),
                'healthy_sc': mean(healthy_sc),
                'healthy_ns': mean(healthy_ns),
                'psa_tests': mean(n_psa_tests_age),
                'n_mri': mean(n_mri_age),
                'n_biopsies': mean(n_biopsies_age),
                'lyrs_healthy_nodiscount': mean(lyrs_healthy_nodiscount_age),
                'lyrs_healthy_discount': mean(lyrs_healthy_discount_age),
                'lyrs_pca_discount': mean(lyrs_pca_discount_age),
                'total_lyrs_discount': mean(lyrs_discount_age),
                'qalys_healthy_discount': mean(qalys_healthy_discount_age),
                'qalys_pca_discount': mean(qalys_pca_discount_age),
                'total_qalys_discount': mean(qalys_discount_age),
                'cost_psa_testing_discount': mean(cost_psa_testing_discount_age),
                'cost_biopsy_discount': mean(cost_biopsy_discount_age),
                'cost_mri_discount': mean(cost_mri_discount_age),
                'cost_staging_discount': mean(cost_staging_discount_age),
                'cost_treatment_discount': mean(cost_tx_discount_age),
                'costs_eol_discount': mean(cost_eol_discount_age),
                'total_cost_discount': mean(cost_discount_age)
            })

            # Totals for each age cohort
            outcomes = pd.DataFrame({
                'cohort_age_at_start': [year],
                'pca_cases': [total(cases_age)],
                'pca_cases_sc': [total(pca_incidence_sc)],
                'pca_cases_ns': [total(pca_incidence_ns)],
                'screen-detected cases': [total(pca_incidence_screened)],
                'post-screening cases': [total(pca_incidence_post_screening)],
                'overdiagnosis': [total(overdiagnosis_age)],
                'pca_deaths': [total(deaths_pca_age)],
                'pca_deaths_sc': [total(pca_death_sc)],
                'pca_deaths_ns': [total(pca_death_ns)],
                'deaths_other_causes': [total(deaths_other_age)],
                'deaths_other_sc': [total(deaths_sc_other_age)],
                'deaths_other_ns': [total(deaths_ns_other_age)],
                'lyrs_healthy_discounted': [total(lyrs_healthy_discount_age)],
                'lyrs_pca_discounted': [total(lyrs_pca_discount_age)],
                'lyrs_undiscounted': [total(lyrs_nodiscount_age)],
                'lyrs_discounted': [total(lyrs_discount_age)],
                'qalys_healthy_discounted': [total(qalys_healthy_discount_age)],
                'qalys_pca_discounted': [total(qalys_pca_discount_age)],
                'qalys_undiscounted': [total(qalys_nodiscount_age)],
                'qalys_discounted': [total(qalys_discount_age)],
                'cost_psa_testing_undiscounted': [total(cost_psa_testing_nodiscount_age)],
                'cost_psa_testing_discounted': [total(cost_psa_testing_discount_age)],
                'cost_mri_undiscounted': [total(cost_mri_nodiscount_age)],
                'cost_mri_discounted': [total(cost_mri_discount_age)],
                'cost_biopsy_undiscounted': [total(cost_biopsy_nodiscount_age)],
                'cost_biopsy_discounted': [total(cost_biopsy_discount_age)],
                'cost_staging_undiscounted': [total(cost_staging_nodiscount_age)],
                'cost_staging_discounted': [total(cost_staging_discount_age)],
                'cost_treatment_undiscounted': [total(cost_tx_nodiscount_age)],
                'cost_treatment_discounted': [total(cost_tx_discount_age)],
                'cost_eol_undiscounted': [total(cost_eol_nodiscount_age)],
                'cost_eol_discounted': [total(cost_eol_discount_age)],
                'costs_undiscounted': [total(cost_nodiscount_age)],
                'costs_discounted': [total(cost_discount_age)],
                'n_psa_tests': [total(n_psa_tests_age)],
                'n_mri': [total(n_mri_age)],
                'n_biopsies': [total(n_biopsies_age)],
            })

            self.simulations['qalys'].append(np.sum(qalys_discount_age, axis=1))
            self.simulations['lyrs'].append(np.sum(lyrs_discount_age, axis=1))
            self.simulations['costs'].append(np.sum(cost_discount_age, axis=1))
            self.simulations['pca_deaths'].append(np.sum(deaths_pca_age, axis=1))
            self.simulations['pca_deaths_sc'].append(np.sum(pca_death_sc, axis=1))
            self.simulations['pca_deaths_ns'].append(np.sum(pca_death_ns, axis=1))
            self.simulations['deaths_other'].append(np.sum(deaths_other_age, axis=1))
            self.simulations['deaths_other_sc'].append(np.sum(deaths_sc_other_age, axis=1))
            self.simulations['deaths_other_ns'].append(np.sum(deaths_ns_other_age, axis=1))
            self.simulations['pca_cases'].append(np.sum(cases_age, axis=1))
            self.simulations['pca_cases_sc'].append(np.sum(pca_incidence_sc, axis=1))
            self.simulations['pca_cases_ns'].append(np.sum(pca_incidence_ns, axis=1))
            self.simulations['screen_detected_cases'].append(np.sum(pca_incidence_screened, axis=1))
            self.simulations['post_screening_cases'].append(np.sum(pca_incidence_post_screening, axis=1))
            self.simulations['cost_mri'].append(np.sum(cost_mri_discount_age, axis=1))
            self.simulations['cost_biopsy'].append(np.sum(cost_biopsy_discount_age, axis=1))
            self.simulations['n_mri'].append(np.sum(n_mri_age, axis=1))
            self.simulations['n_biopsies'].append(np.sum(n_biopsies_age, axis=1))
            self.simulations['n_psa_tests'].append(np.sum(n_psa_tests_age, axis=1))
            self.simulations['overdiagnosis'].append(np.sum(overdiagnosis_age, axis=1))


            self.cohort_list.append(cohort)
            self.outcomes_list.append(outcomes)

        return self.cohort_list, self.outcomes_list, self.simulations
