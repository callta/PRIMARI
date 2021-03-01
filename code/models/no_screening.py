import pandas as pd
import numpy as np
from collections import defaultdict
from utils.save_models import Save
from utils.functions import mean, total


class NoScreening(Save):

    """Build cohort of non-screened using 2019 guidelines"""

    def __init__(self, params):
        self.cohort_list = []
        self.outcomes_list = []
        self.simulations = defaultdict(list)
        self.run_model = self._run_model(params)

    def _run_model(self, params):

        # Loop through age cohorts
        for year in np.arange(55, 70):

            incidence = params.adjusted_incidence[:, year-45:]
            pca_mortality = params.adjusted_mortality[:, year-45:]
            mortality_other_causes = params.death_other_causes[:, year-45:]
            localised_stage = params.ns_localised_stage[:, year-45:]
            advanced_stage = params.ns_advanced_stage[:, year-45:]
            tx_costs_local = params.tx_costs * params.tx.localised.values
            tx_costs_adv = params.tx_costs * params.tx.advanced.values

            # Year 1 in the model
            #####################
            age = np.arange(year, 90)
            length_df = len(age)

            # Cohorts, numbers 'healthy', and incident cases
            cohort = np.array([np.repeat(params.pop.loc[year, :], length_df)] * params.sims)
            pca_alive = np.array([np.zeros(length_df)] * params.sims)
            healthy = cohort - pca_alive
            pca_incidence = healthy * incidence

            # Deaths
            pca_death = ((pca_alive * pca_mortality)
                         + (healthy * pca_mortality))

            pca_death_other = ((pca_incidence
                                + pca_alive
                                - pca_death)
                               * mortality_other_causes)

            healthy_death_other = ((healthy - pca_incidence)
                                   * mortality_other_causes)

            total_death = (pca_death
                           + pca_death_other
                           + healthy_death_other)

            # Prevalent cases & life-years
            pca_prevalence_ns = (pca_incidence
                                 - pca_death
                                 - pca_death_other)

            lyrs_pca_nodiscount = pca_prevalence_ns * 0.5

            # Treatment costs
            costs_tx = np.array([np.zeros(length_df)] * params.sims)

            costs_tx[:, 0] = ((pca_incidence[:, 0]
                               * localised_stage[:, 0].T
                               * tx_costs_local.T).sum(axis=0)

                              + (pca_incidence[:, 0]
                                 * advanced_stage[:, 0].T
                                 * tx_costs_adv.T).sum(axis=0)

                              * params.relative_cost_clinically_detected[:, 0])

            # Year 2 onwards
            ################
            total_cycles = length_df
            for i in range(1, total_cycles):

               # Cohorts, numbers 'healthy', and incident cases
                cohort[:, i] = cohort[:, i-1] - total_death[:, i-1]

                pca_alive[:, i] = (pca_alive[:, i-1]
                                   + pca_incidence[:, i-1]
                                   - pca_death[:, i-1]
                                   - pca_death_other[:, i-1]) # PCa alive at the beginning of the year

                healthy[:, i] = cohort[:, i] - pca_alive[:, i]

                pca_incidence[:, i] = healthy[:, i] * incidence[:, i]

                # Deaths
                pca_death[:, i] = ((pca_alive[:, i] * pca_mortality[:, i])
                                   + (healthy[:, i] * pca_mortality[:, i]))

                pca_death_other[:, i] = ((pca_incidence[:, i]
                                          + pca_alive[:, i]
                                          - pca_death[:, i])
                                         * mortality_other_causes[:, i])

                healthy_death_other[:, i] = ((healthy[:, i] - pca_incidence[:, i])
                                             * mortality_other_causes[:, i])

                total_death[:, i] = (pca_death[:, i]
                                     + pca_death_other[:, i]
                                     + healthy_death_other[:, i])

                # Prevalent cases & life-years
                pca_prevalence_ns[:, i] = (pca_incidence[:, i]
                                           + pca_alive[:, i]
                                           - pca_death[:, i]
                                           - pca_death_other[:, i])

                lyrs_pca_nodiscount[:, i] = ((pca_prevalence_ns[:, i-1]
                                              + pca_prevalence_ns[:, i])
                                             * 0.5)

                # Costs
                costs_tx[:, i] = ((pca_incidence[:, i]
                                  * localised_stage[:, i].T
                                  * tx_costs_local.T).sum(axis=0)

                                  + (pca_incidence[:, i]
                                     * advanced_stage[:, i].T
                                     * tx_costs_adv.T).sum(axis=0)

                                  * params.relative_cost_clinically_detected[:, i])

            ##############
            # Life-years #
            ##############

            # Life-years ('healthy')
            lyrs_healthy_nodiscount_ns = healthy - (0.5*(healthy_death_other+pca_incidence))
            lyrs_healthy_discount_ns = lyrs_healthy_nodiscount_ns * params.discount_factor[:total_cycles]

            # Life-years with prostate cancer
            lyrs_pca_discount_ns = lyrs_pca_nodiscount * params.discount_factor[:total_cycles]

            # Total life-years
            ##################
            lyrs_nodiscount_ns = lyrs_healthy_nodiscount_ns + lyrs_pca_nodiscount
            lyrs_discount_ns = lyrs_healthy_discount_ns + lyrs_pca_discount_ns

            #########
            # QALYs #
            #########

            # QALYs in the healthy
            qalys_healthy_nodiscount_ns = lyrs_healthy_nodiscount_ns * params.utility_background[:, year-45:]
            qalys_healthy_discount_ns = lyrs_healthy_discount_ns * params.utility_background[:, year-45:]

            # QALYs with prostate cancer
            qalys_pca_nodiscount_ns = lyrs_pca_nodiscount * params.utility_pca[:, year-45:]
            qalys_pca_discount_ns = lyrs_pca_discount_ns * params.utility_pca[:, year-45:]

            # Total QALYs
            #############
            qalys_nodiscount_ns = qalys_healthy_nodiscount_ns + qalys_pca_nodiscount_ns
            qalys_discount_ns = qalys_healthy_discount_ns + qalys_pca_discount_ns

            ###############
            # PSA testing #
            ###############

            # Cost of PSA testing
            n_psa_tests_ns = ((pca_incidence/params.p_biopsy_ns[:, year-45:])
                              * params.n_psa_tests[:, year-45:])

            cost_psa_testing_nodiscount_ns = (n_psa_tests_ns
                                              * params.cost_psa[:, year-45:]
                                              * params.relative_cost_clinically_detected[:, year-45:])

            cost_psa_testing_discount_ns = cost_psa_testing_nodiscount_ns * params.discount_factor[:total_cycles]

            #######
            # MRI #
            #######

            # Number and costs of MRI
            n_mri_ns = pca_incidence / params.p_biopsy_ns[:, year-45:]

            cost_mri_nodiscount_ns = (n_mri_ns
                                      * np.array([params.cost_mri]).T
                                      * params.relative_cost_clinically_detected[:, year-45:])

            cost_mri_discount_ns = cost_mri_nodiscount_ns * params.discount_factor[:total_cycles]

            ############
            # Biopsies #
            ############

            # Cost of suspected cancer - biopsies
            n_biopsies_ns = n_mri_ns * params.n_biopsies_primari[:, year-45:]

            cost_biopsy_nodiscount_ns = (n_biopsies_ns
                                         * params.cost_biopsy[:, year-45:]
                                         * params.relative_cost_clinically_detected[:, year-45:])

            cost_biopsy_discount_ns = cost_biopsy_nodiscount_ns * params.discount_factor[:total_cycles]

            ###########
            # Staging #
            ###########

            # Cost of staging
            cost_staging_nodiscount_ns = (params.cost_assessment
                                          * advanced_stage.T
                                          * pca_incidence.T
                                          * params.relative_cost_clinically_detected[:, year-45:].T).T

            cost_staging_discount_ns = cost_staging_nodiscount_ns * params.discount_factor[:total_cycles]

            #############
            # EOL Costs #
            #############

            # Cost in last 12 months of life
            cost_eol_nodiscount_ns = (params.cost_pca_death * pca_death.T).T
            cost_eol_discount_ns = cost_eol_nodiscount_ns * params.discount_factor[:total_cycles]

            ###################
            # Treatment costs #
            ###################

            # Costs of treatment
            cost_tx_discount_ns = costs_tx * params.discount_factor[:total_cycles]

            # Amalgamated costs
            cost_nodiscount_ns = (cost_psa_testing_nodiscount_ns
                                  + cost_mri_nodiscount_ns
                                  + cost_biopsy_nodiscount_ns
                                  + cost_staging_nodiscount_ns
                                  + costs_tx
                                  + cost_eol_nodiscount_ns)

            cost_discount_ns = (cost_psa_testing_discount_ns
                                + cost_mri_discount_ns
                                + cost_biopsy_discount_ns
                                + cost_staging_discount_ns
                                + cost_tx_discount_ns
                                + cost_eol_discount_ns)

            #######################################
            # Generate dataframes of the outcomes #
            #######################################

            # Generate a mean dataframe for each age cohort
            cohort = pd.DataFrame({
                'age': age,
                'pca_cases': mean(pca_incidence),
                'deaths_other': mean((pca_death_other+healthy_death_other)),
                'deaths_pca': mean(pca_death),
                'pca_alive': mean(pca_alive),
                'healthy': mean(healthy),
                'psa_tests': mean(n_psa_tests_ns),
                'n_mri': mean(n_mri_ns),
                'n_biopsies': mean(n_biopsies_ns),
                'lyrs_healthy_nodiscount': mean(lyrs_healthy_nodiscount_ns),
                'lyrs_healthy_discount': mean(lyrs_healthy_discount_ns),
                'lyrs_pca_discount': mean(lyrs_pca_discount_ns),
                'total_lyrs_discount': mean(lyrs_discount_ns),
                'qalys_healthy_discount': mean(qalys_healthy_discount_ns),
                'qalys_pca_discount': mean(qalys_pca_discount_ns),
                'total_qalys_discount': mean(qalys_discount_ns),
                'cost_psa_testing_discount': mean(cost_psa_testing_discount_ns),
                'cost_mri_discount': mean(cost_mri_discount_ns),
                'cost_biopsy_discount': mean(cost_biopsy_discount_ns),
                'cost_staging_discount': mean(cost_staging_discount_ns),
                'cost_treatment_discount': mean(cost_tx_discount_ns),
                'costs_eol_discount': mean(cost_eol_discount_ns),
                'total_cost_discount': mean(cost_discount_ns)
            })

            # Totals for each age cohort
            outcomes = pd.DataFrame({
                'cohort_age_at_start': [year],
                'pca_cases': [total(pca_incidence)],
                'pca_deaths': [total(pca_death)],
                'deaths_other_causes': [total((pca_death_other+healthy_death_other))],
                'lyrs_healthy_discounted': [total(lyrs_healthy_discount_ns)],
                'lyrs_pca_discounted': [total(lyrs_pca_discount_ns)],
                'lyrs_undiscounted': [total(lyrs_nodiscount_ns)],
                'lyrs_discounted': [total(lyrs_discount_ns)],
                'qalys_healthy_discounted': [total(qalys_healthy_discount_ns)],
                'qalys_pca_discounted': [total(qalys_pca_discount_ns)],
                'qalys_undiscounted': [total(qalys_nodiscount_ns)],
                'qalys_discounted': [total(qalys_discount_ns)],
                'cost_psa_testing_undiscounted': [total(cost_psa_testing_nodiscount_ns)],
                'cost_psa_testing_discounted': [total(cost_psa_testing_discount_ns)],
                'cost_mri_undiscounted': [total(cost_mri_nodiscount_ns)],
                'cost_mri_discounted': [total(cost_mri_discount_ns)],
                'cost_biopsy_undiscounted': [total(cost_biopsy_nodiscount_ns)],
                'cost_biopsy_discounted': [total(cost_biopsy_discount_ns)],
                'cost_staging_undiscounted': [total(cost_staging_nodiscount_ns)],
                'cost_staging_discounted': [total(cost_staging_discount_ns)],
                'cost_eol_undiscounted': [total(cost_eol_nodiscount_ns)],
                'cost_eol_discounted': [total(cost_eol_discount_ns)],
                'cost_treatment_undiscounted': [total(costs_tx)],
                'cost_treatment_discounted': [total(cost_tx_discount_ns)],
                'costs_undiscounted': [total(cost_nodiscount_ns)],
                'costs_discounted': [total(cost_discount_ns)],
                'n_psa_tests': [total(n_psa_tests_ns)],
                'n_mri': [total(n_mri_ns)],
                'n_biopsies': [total(n_biopsies_ns)],
                'overdiagnosis': [0]
            })

            self.simulations['qalys'].append(np.sum(qalys_discount_ns, axis=1))
            self.simulations['lyrs'].append(np.sum(lyrs_discount_ns, axis=1))
            self.simulations['costs'].append(np.sum(cost_discount_ns, axis=1))
            self.simulations['pca_deaths'].append(np.sum(pca_death, axis=1))
            self.simulations['pca_cases'].append(np.sum(pca_incidence, axis=1))
            self.simulations['cost_mri'].append(np.sum(cost_mri_discount_ns, axis=1))
            self.simulations['cost_biopsy'].append(np.sum(cost_biopsy_discount_ns, axis=1))
            self.simulations['n_mri'].append(np.sum(n_mri_ns, axis=1))
            self.simulations['n_biopsies'].append(np.sum(n_biopsies_ns, axis=1))
            self.simulations['n_psa_tests'].append(np.sum(n_psa_tests_ns, axis=1))
            self.simulations['overdiagnosis'].append([0])

            self.cohort_list.append(cohort)
            self.outcomes_list.append(outcomes)

        return self.cohort_list, self.outcomes_list, self.simulations
