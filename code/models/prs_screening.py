import pandas as pd
import numpy as np
from collections import defaultdict
from utils.save_models import Save
from utils.functions import mean, total


class PrsScreening(Save):

    """Build cohort of risk-tailored screening using MRI-first approach
    to diagnosis, following 2019 NICE guidelines."""

    def __init__(self, params, a_risk, od_by_risk:bool=False):
        self.cohort_list = []
        self.outcomes_list = []
        self.simulations = defaultdict(list)
        self.run_model = self._run_model(params, a_risk, od_by_risk)

    def _run_model(self, params, a_risk, od_by_risk):

        # Loop through age cohorts
        for year in np.arange(55, 70):

            # PCa incidence
            pca_incidence_prs = params.adjusted_incidence.copy()
            pca_incidence_prs[:,10:25] = (pca_incidence_prs[:,10:25].T * params.rr_incidence_adjusted[year-45,:]).T
            pca_incidence_prs[:,25:35] = pca_incidence_prs[:,25:35] * np.linspace(params.post_sc_incidence_drop,1,10)
            pca_incidence_prs = pca_incidence_prs[:, year-45:]

            # Death from PCa
            pca_mortality_prs = params.adjusted_mortality.copy()
            pca_mortality_prs[:,10:15] = pca_mortality_prs[:,10:15] * np.linspace(1,0.8,5)
            pca_mortality_prs[:,15:] = pca_mortality_prs[:,15:] * params.rr_death_screening[:,15:]
            pca_mortality_prs = pca_mortality_prs[:, year-45:]

            # Probability of being screened
            p_screened = np.array(params.uptake_prs * a_risk.loc[year,'p_above_threshold'])
            p_ns = np.array((1-params.uptake_prs) * a_risk.loc[year,'p_above_threshold'])
            p_nos = np.array(params.compliance * (1-a_risk.loc[year,'p_above_threshold']))
            p_nos_screened = np.array((1-params.compliance) * (1-a_risk.loc[year,'p_above_threshold']))

            if year < 55:
                # PCa incidence
                p_pca_screened = params.adjusted_incidence[:, year-45:]
                p_pca_ns = params.adjusted_incidence[:, year-45:]
                p_pca_nos = params.adjusted_incidence[:, year-45:]
                p_pca_nos_screened = params.adjusted_incidence[:, year-45:]

                # Death from PCa
                p_pca_death_screened = params.adjusted_mortality[:, year-45:]
                p_pca_death_ns = params.adjusted_mortality[:, year-45:]
                p_pca_death_nos = params.adjusted_mortality[:, year-45:]
                p_pca_death_nos_screened = params.adjusted_mortality[:, year-45:]

                # Proportion of cancers detected by screening at a localised / advanced stage
                advanced_stage_sc = params.ns_advanced_stage[:, year-45:]
                advanced_stage_ns = params.ns_advanced_stage[:, year-45:]
                advanced_stage_nos_sc = params.ns_advanced_stage[:, year-45:]
                advanced_stage_nos = params.ns_advanced_stage[:, year-45:]

                localised_stage_sc = params.ns_localised_stage[:, year-45:]
                localised_stage_ns = params.ns_localised_stage[:, year-45:]
                localised_stage_nos_sc = params.ns_localised_stage[:, year-45:]
                localised_stage_nos = params.ns_localised_stage[:, year-45:]

            elif year > 54:
                # PCa incidence
                p_pca_screened = pca_incidence_prs * a_risk.loc[year, 'rr_high']
                p_pca_ns = params.adjusted_incidence[:, year-45:] * a_risk.loc[year,'rr_high']
                p_pca_nos = params.adjusted_incidence[:, year-45:] * a_risk.loc[year,'rr_low']
                p_pca_nos_screened = pca_incidence_prs * a_risk.loc[year,'rr_low']

                # Death from PCa
                p_pca_death_screened = pca_mortality_prs * a_risk.loc[year,'rr_high']
                p_pca_death_ns = params.adjusted_mortality[:, year-45:] * a_risk.loc[year,'rr_high']
                p_pca_death_nos = params.adjusted_mortality[:, year-45:] * a_risk.loc[year,'rr_low']
                p_pca_death_nos_screened = pca_mortality_prs * a_risk.loc[year,'rr_low']

                # Proportion of cancers detected by screening at a localised / advanced stage
                advanced_stage_sc = params.sc_advanced_stage[:, year-45:]
                localised_stage_sc = params.sc_localised_stage[:, year-45:]

                advanced_stage_ns = params.ns_advanced_stage[:, year-45:]
                localised_stage_ns = params.ns_localised_stage[:, year-45:]

                advanced_stage_nos_sc = params.sc_advanced_stage[:, year-45:]
                localised_stage_nos_sc = params.sc_localised_stage[:, year-45:]

                advanced_stage_nos = params.ns_advanced_stage[:, year-45:]
                localised_stage_nos = params.ns_localised_stage[:, year-45:]

                # Proportion misclassified with MRI-first approach
                p_misclassified = params.misclassified[:, year-45:]

            mortality_other_causes = params.death_other_causes[:, year-45:]
            tx_costs_local = params.tx_costs * params.tx.localised.values
            tx_costs_adv = params.tx_costs * params.tx.advanced.values

            #####################
            # Year 1 in the model
            #####################
            age = np.arange(year,90)
            length_df = len(age)
            length_screen = len(np.arange(year,70)) # number of screening years depending on age cohort starting

            # Cohorts, numbers 'healthy', and incident cases
            cohort_sc = np.array([np.repeat(params.pop.loc[year, :], length_df)]*params.sims) * p_screened
            cohort_ns = np.array([np.repeat(params.pop.loc[year, :], length_df)]*params.sims) * p_ns
            cohort_nos = np.array([np.repeat(params.pop.loc[year, :], length_df)]*params.sims) * p_nos
            cohort_nos_sc = np.array([np.repeat(params.pop.loc[year, :], length_df)]*params.sims) * p_nos_screened

            pca_alive_sc = np.array([np.zeros(length_df)]*params.sims)
            pca_alive_ns = np.array([np.zeros(length_df)]*params.sims)
            pca_alive_nos = np.array([np.zeros(length_df)]*params.sims)
            pca_alive_nos_sc = np.array([np.zeros(length_df)]*params.sims)

            healthy_sc = cohort_sc - pca_alive_sc
            healthy_ns = cohort_ns - pca_alive_ns
            healthy_nos = cohort_nos - pca_alive_nos
            healthy_nos_sc = cohort_nos_sc - pca_alive_nos_sc

            pca_incidence_sc = healthy_sc * p_pca_screened
            pca_incidence_nos_sc = healthy_nos_sc * p_pca_nos_screened

            if year > 54:
                pca_incidence_screened = pca_incidence_sc.copy()  # Screen-detected cancers
                pca_misclassified = pca_incidence_screened * p_misclassified
                pca_incidence_post_screening = np.array([np.zeros(length_df)]*params.sims)  # Post-screening cancers - 0 until model reaches age 70.

                pca_incidence_nos_sc_screened = pca_incidence_nos_sc.copy()  # Screen-detected cancers
                pca_misclassified_nos_sc = pca_incidence_nos_sc_screened * p_misclassified  # Misclassified amongst those screened despite not being eligible
                pca_incidence_nos_sc_post_screening = np.array([np.zeros(length_df)]*params.sims)  # Post-screening cancers - 0 until model reaches age 70.

            elif year < 55:
                # Zero as no screening in any of these cohorts
                pca_incidence_screened = np.array([np.zeros(length_df)]*params.sims)
                pca_misclassified = np.array([np.zeros(length_df)]*params.sims)
                pca_incidence_post_screening = np.array([np.zeros(length_df)]*params.sims)

                pca_incidence_nos_sc_screened = np.array([np.zeros(length_df)]*params.sims)
                pca_misclassified_nos_sc = np.array([np.zeros(length_df)]*params.sims)
                pca_incidence_nos_sc_post_screening = np.array([np.zeros(length_df)]*params.sims)

            pca_incidence_ns = healthy_ns * p_pca_ns
            pca_incidence_nos = healthy_nos * p_pca_nos

            # Misclassified will become clinically detected over params.mst
            pca_misclassified_array = np.array([np.zeros(length_df)] * params.sims)
            n_misclassified = np.array([pca_misclassified[:, 0]/params.mst]*params.mst).T
            pca_misclassified_array[:, 1:1+params.mst] = n_misclassified

            pca_misclassified_array_nos_sc = np.array([np.zeros(length_df)] * params.sims)
            n_misclassified_nos_sc = np.array([pca_misclassified_nos_sc[:, 0]/params.mst]*params.mst).T
            pca_misclassified_array_nos_sc[:, 1:1+params.mst] = n_misclassified_nos_sc

            # Deaths
            pca_death_sc = ((pca_alive_sc * p_pca_death_screened)
                            + (healthy_sc * p_pca_death_screened))

            pca_death_ns = ((pca_alive_ns * p_pca_death_ns)
                            + (healthy_ns * p_pca_death_ns))

            pca_death_nos = ((pca_alive_nos * p_pca_death_nos)
                             + (healthy_nos * p_pca_death_nos))

            pca_death_nos_sc = ((pca_alive_nos_sc * p_pca_death_nos_screened)
                                + (healthy_nos_sc * p_pca_death_nos_screened))

            pca_death_other_sc = ((pca_incidence_sc
                                  + pca_alive_sc
                                  - pca_death_sc)
                                 * mortality_other_causes)

            pca_death_other_ns = ((pca_incidence_ns
                                  + pca_alive_ns
                                  - pca_death_ns)
                                 * mortality_other_causes)

            pca_death_other_nos = ((pca_incidence_nos
                                    + pca_alive_nos
                                    - pca_death_nos)
                                   * mortality_other_causes)

            pca_death_other_nos_sc  = ((pca_incidence_nos_sc
                                       + pca_alive_nos_sc
                                       - pca_death_nos_sc)
                                      * mortality_other_causes)

            healthy_death_other_sc = (healthy_sc-pca_incidence_sc) * mortality_other_causes
            healthy_death_other_ns = (healthy_ns-pca_incidence_ns) * mortality_other_causes
            healthy_death_other_nos = (healthy_nos-pca_incidence_nos) * mortality_other_causes
            healthy_death_other_nos_sc = (healthy_nos_sc-pca_incidence_nos_sc) * mortality_other_causes

            total_death_sc = (pca_death_sc
                              + pca_death_other_sc
                              + healthy_death_other_sc)

            total_death_ns = (pca_death_ns
                              + pca_death_other_ns
                              + healthy_death_other_ns)

            total_death_nos = (pca_death_nos
                               + pca_death_other_nos
                               + healthy_death_other_nos)

            total_death_nos_sc = (pca_death_nos_sc
                                  + pca_death_other_nos_sc
                                  + healthy_death_other_nos_sc)

            total_death = (total_death_sc
                           + total_death_ns
                           + total_death_nos
                           + total_death_nos_sc)

            # Prevalent cases & life-years
            pca_prevalence_sc = (pca_incidence_sc
                                 - pca_death_sc
                                 - pca_death_other_sc)

            pca_prevalence_ns = (pca_incidence_ns
                                 - pca_death_ns
                                 - pca_death_other_ns)

            pca_prevalence_nos = (pca_incidence_nos
                                  - pca_death_nos
                                  - pca_death_other_nos)

            pca_prevalence_nos_sc = (pca_incidence_nos_sc
                                     - pca_death_nos_sc
                                     - pca_death_other_nos_sc)

            lyrs_pca_sc_nodiscount = pca_prevalence_sc * 0.5
            lyrs_pca_ns_nodiscount = pca_prevalence_ns * 0.5
            lyrs_pca_nos_nodiscount = pca_prevalence_nos * 0.5
            lyrs_pca_nos_sc_nodiscount = pca_prevalence_nos_sc * 0.5

            # Costs
            if year > 54:
                costs_tx_sc = np.array([np.zeros(length_df)]*params.sims)
                costs_tx_screened = np.array([np.zeros(length_df)]*params.sims)
                costs_tx_post_screening = np.array([np.zeros(length_df)]*params.sims)

                costs_tx_screened[:, 0] = ((pca_incidence_screened[:, 0]
                                           * localised_stage_sc[:, 0].T
                                           * tx_costs_local.T).sum(axis=0)

                                          + (pca_incidence_screened[:, 0]
                                            * advanced_stage_sc[:, 0].T
                                            * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

                costs_tx_post_screening[:, 0] = ((pca_incidence_post_screening[:, 0]
                                                 * localised_stage_ns[:, 0].T
                                                 * tx_costs_local.T).sum(axis=0)

                                                + (pca_incidence_post_screening[:, 0]
                                                   * advanced_stage_ns[:, 0].T
                                                   * tx_costs_adv.T).sum(axis=0)

                                                * params.relative_cost_clinically_detected[:, 0]) # cost of post-screening cancers

                costs_tx_sc[:, 0] = costs_tx_screened[:, 0] + costs_tx_post_screening[:, 0] # total cost in screened arms

                costs_tx_nos_sc = np.array([np.zeros(length_df)] * params.sims)
                costs_tx_nos_sc_screened = np.array([np.zeros(length_df)] * params.sims)
                costs_tx_nos_sc_post_screening = np.array([np.zeros(length_df)] * params.sims)

                costs_tx_nos_sc_screened[:, 0] = ((pca_incidence_nos_sc_screened[:, 0]
                                                  * localised_stage_nos_sc[:, 0].T
                                                  * tx_costs_local.T).sum(axis=0)

                                                 + (pca_incidence_nos_sc_screened[:, 0]
                                                    * advanced_stage_nos_sc[:, 0].T
                                                    * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

                costs_tx_nos_sc_post_screening[:, 0] = ((pca_incidence_nos_sc_post_screening[:, 0]
                                                        * localised_stage_nos[:, 0].T
                                                        * tx_costs_local.T).sum(axis=0)

                                                       + (pca_incidence_nos_sc_post_screening[:, 0]
                                                          * advanced_stage_nos[:, 0].T
                                                          * tx_costs_adv.T).sum(axis=0)

                                                       * params.relative_cost_clinically_detected[:, 0]) # cost of post-screening cancers

                costs_tx_nos_sc[:, 0] = costs_tx_nos_sc_screened[:, 0] + costs_tx_nos_sc_post_screening[:, 0] # total cost in screened arms

            elif year < 55:
                costs_tx_sc = np.array([np.zeros(length_df)] * params.sims)
                costs_tx_sc[:, 0] = ((pca_incidence_sc[:, 0]
                                     * localised_stage_ns[:, 0].T # not actually receiving screening (<55) so rr_adv_screening doesn't apply
                                     * tx_costs_local.T).sum(axis=0)

                                    + (pca_incidence_sc[:, 0]
                                       * advanced_stage_ns[:, 0].T
                                       * tx_costs_adv.T).sum(axis=0)

                                   * params.relative_cost_clinically_detected[:, 0])

                costs_tx_nos_sc = np.array([np.zeros(length_df)] * params.sims)
                costs_tx_nos_sc[:, 0] = ((pca_incidence_nos_sc[:, 0]
                                         * localised_stage_nos[:, 0].T
                                         * tx_costs_local.T).sum(axis=0)

                                        + (pca_incidence_nos_sc[:, 0]
                                         * advanced_stage_nos[:, 0].T
                                         * tx_costs_adv.T).sum(axis=0)

                                       * params.relative_cost_clinically_detected[:, 0])

            costs_tx_ns = np.array([np.zeros(length_df)] * params.sims)
            costs_tx_ns[:, 0] = ((pca_incidence_ns[:, 0]
                                 * localised_stage_ns[:, 0].T
                                 * tx_costs_local.T).sum(axis=0)

                                + (pca_incidence_ns[:, 0]
                                 * advanced_stage_ns[:, 0].T
                                 * tx_costs_adv.T).sum(axis=0)

                               * params.relative_cost_clinically_detected[:, 0])

            costs_tx_nos = np.array([np.zeros(length_df)] * params.sims)
            costs_tx_nos[:, 0] = ((pca_incidence_nos[:, 0]
                                  * localised_stage_nos[:, 0].T
                                  * tx_costs_local.T).sum(axis=0)

                                 + (pca_incidence_nos[:, 0]
                                  * advanced_stage_nos[:, 0].T
                                  * tx_costs_adv.T).sum(axis=0)

                                * params.relative_cost_clinically_detected[:, 0])

            # Year 2 onwards
            ################
            total_cycles = length_df
            for i in range(1, total_cycles):

               # Cohorts, numbers 'healthy', incident & prevalent cases
                cohort_sc[:, i] = cohort_sc[:, i-1] - total_death_sc[:, i-1]
                cohort_ns[:, i] = cohort_ns[:, i-1] - total_death_ns[:, i-1]
                cohort_nos[:, i] = cohort_nos[:, i-1] - total_death_nos[:, i-1]
                cohort_nos_sc[:, i] = cohort_nos_sc[:, i-1] - total_death_nos_sc[:, i-1]

                pca_alive_sc[:, i] = (pca_alive_sc[:, i-1]
                                     + pca_incidence_sc[:, i-1]
                                     - pca_death_sc[:, i-1]
                                     - pca_death_other_sc[:, i-1])

                pca_alive_ns[:, i] = (pca_alive_ns[:, i-1]
                                     + pca_incidence_ns[:, i-1]
                                     - pca_death_ns[:, i-1]
                                     - pca_death_other_ns[:, i-1])

                pca_alive_nos[:, i] = (pca_alive_nos[:, i-1]
                                      + pca_incidence_nos[:, i-1]
                                      - pca_death_nos[:, i-1]
                                      - pca_death_other_nos[:, i-1])

                pca_alive_nos_sc[:, i] = (pca_alive_nos_sc[:, i-1]
                                         + pca_incidence_nos_sc[:, i-1]
                                         - pca_death_nos_sc[:, i-1]
                                         - pca_death_other_nos_sc[:, i-1])

                healthy_sc[:, i] = cohort_sc[:, i] - pca_alive_sc[:, i]
                healthy_ns[:, i] = cohort_ns[:, i] - pca_alive_ns[:, i]
                healthy_nos[:, i] = cohort_nos[:, i] - pca_alive_nos[:, i]
                healthy_nos_sc[:, i] = cohort_nos_sc[:, i] - pca_alive_nos_sc[:, i]

                pca_incidence_sc[:, i] = (healthy_sc[:, i] * p_pca_screened[:, i]) + pca_misclassified_array[:, i]
                pca_incidence_nos_sc[:, i] = (healthy_nos_sc[:, i] * p_pca_nos_screened[:, i]) + pca_misclassified_array_nos_sc[:, i]

                if year > 54:
                    if i < length_screen:
                        pca_incidence_screened[:, i] = healthy_sc[:, i] * p_pca_screened[:, i]
                        pca_incidence_post_screening[:, i] = 0

                        pca_incidence_nos_sc_screened[:, i] = healthy_nos_sc[:, i] * p_pca_nos_screened[:, i]
                        pca_incidence_nos_sc_post_screening[:, i] = 0

                        # Misclassified
                        pca_misclassified[:, i] = pca_incidence_screened[:, i] * p_misclassified[:, i]
                        n_misclassified = np.array([pca_misclassified[:, i]/params.mst]*params.mst).T # Need array to be same size as pca_misclassified_array to add
                        pca_misclassified_array[:, i+1:i+1+params.mst] = pca_misclassified_array[:, i+1:i+1+params.mst] + n_misclassified

                        pca_misclassified_nos_sc[:, i] = pca_incidence_nos_sc_screened[:, i] * p_misclassified[:, i]
                        n_misclassified_nos_sc = np.array([pca_misclassified_nos_sc[:, i]/params.mst]*params.mst).T
                        pca_misclassified_array_nos_sc[:, i+1:i+1+params.mst] = pca_misclassified_array_nos_sc[:, i+1:i+1+params.mst] + n_misclassified_nos_sc

                    else:
                        pca_incidence_screened[:, i] = 0
                        pca_incidence_post_screening[:, i] = healthy_sc[:, i] * p_pca_screened[:, i]

                        pca_incidence_nos_sc_screened[:, i] = 0
                        pca_incidence_nos_sc_post_screening[:, i] = healthy_nos_sc[:, i] * p_pca_nos_screened[:, i]

                        pca_misclassified[:, i] = 0
                        pca_misclassified_nos_sc[:, i] = 0

                elif year < 55:
                    pca_incidence_screened[:, i] = 0
                    pca_incidence_post_screening[:, i] = 0

                    pca_incidence_nos_sc_screened[:, i] = 0
                    pca_incidence_nos_sc_post_screening[:, i] = 0

                    pca_misclassified[:, i] = 0
                    pca_misclassified_nos_sc[:, i] = 0

                pca_incidence_ns[:, i] = healthy_ns[:, i] * p_pca_ns[:, i]
                pca_incidence_nos[:, i] = healthy_nos[:, i] * p_pca_nos[:, i]

                # Deaths
                pca_death_sc[:, i] = ((pca_alive_sc[:, i]*p_pca_death_screened[:, i])
                                     + (healthy_sc[:, i]*p_pca_death_screened[:, i]))

                pca_death_ns[:, i] = ((pca_alive_ns[:, i]*p_pca_death_ns[:, i])
                                     + (healthy_ns[:, i]*p_pca_death_ns[:, i]))

                pca_death_nos[:, i] = ((pca_alive_nos[:, i]*p_pca_death_nos[:, i])
                                      + (healthy_nos[:, i]*p_pca_death_nos[:, i]))

                pca_death_nos_sc[:, i] = ((pca_alive_nos_sc[:, i]*p_pca_death_nos_screened[:, i])
                                         + (healthy_nos_sc[:, i]*p_pca_death_nos_screened[:, i]))

                pca_death_other_sc[:, i] = ((pca_incidence_sc[:, i]
                                            + pca_alive_sc[:, i]
                                            - pca_death_sc[:, i])
                                           * mortality_other_causes[:, i])

                pca_death_other_ns[:, i] = ((pca_incidence_ns[:, i]
                                            + pca_alive_ns[:, i]
                                            - pca_death_ns[:, i])
                                           * mortality_other_causes[:, i])

                pca_death_other_nos[:, i] = ((pca_incidence_nos[:, i]
                                             + pca_alive_nos[:, i]
                                             - pca_death_nos[:, i])
                                            * mortality_other_causes[:, i])

                pca_death_other_nos_sc[:, i] = ((pca_incidence_nos_sc[:, i]
                                                + pca_alive_nos_sc[:, i]
                                                - pca_death_nos_sc[:, i])
                                               * mortality_other_causes[:, i])

                healthy_death_other_sc[:, i] = ((healthy_sc[:, i]-pca_incidence_sc[:, i])
                                               * mortality_other_causes[:, i])

                healthy_death_other_ns[:, i] = ((healthy_ns[:, i]-pca_incidence_ns[:, i])
                                               * mortality_other_causes[:, i])

                healthy_death_other_nos[:, i] = ((healthy_nos[:, i]-pca_incidence_nos[:, i])
                                                * mortality_other_causes[:, i])

                healthy_death_other_nos_sc[:, i] = ((healthy_nos_sc[:, i]-pca_incidence_nos_sc[:, i])
                                                   * mortality_other_causes[:, i])

                total_death_sc[:, i] = (pca_death_sc[:, i]
                                       + pca_death_other_sc[:, i]
                                       + healthy_death_other_sc[:, i])

                total_death_ns[:, i] = (pca_death_ns[:, i]
                                       + pca_death_other_ns[:, i]
                                       + healthy_death_other_ns[:, i])

                total_death_nos[:, i] = (pca_death_nos[:, i]
                                        + pca_death_other_nos[:, i]
                                        + healthy_death_other_nos[:, i])

                total_death_nos_sc[:, i] = (pca_death_nos_sc[:, i]
                                           + pca_death_other_nos_sc[:, i]
                                           + healthy_death_other_nos_sc[:, i])

                total_death[:, i] = (total_death_sc[:, i]
                                    + total_death_ns[:, i]
                                    + total_death_nos[:, i]
                                    + total_death_nos_sc[:, i])

                # Prevalent cases & life-years
                pca_prevalence_sc[:, i] = (pca_incidence_sc[:, i]
                                          + pca_alive_sc[:, i]
                                          - pca_death_sc[:, i]
                                          - pca_death_other_sc[:, i])

                pca_prevalence_ns[:, i] = (pca_incidence_ns[:, i]
                                          + pca_alive_ns[:, i]
                                          - pca_death_ns[:, i]
                                          - pca_death_other_ns[:, i])

                pca_prevalence_nos[:, i] = (pca_incidence_nos[:, i]
                                           + pca_alive_nos[:, i]
                                           - pca_death_nos[:, i]
                                           - pca_death_other_nos[:, i])

                pca_prevalence_nos_sc[:, i] = (pca_incidence_nos_sc[:, i]
                                              + pca_alive_nos_sc[:, i]
                                              - pca_death_nos_sc[:, i]
                                              - pca_death_other_nos_sc[:, i])

                lyrs_pca_sc_nodiscount[:, i] = ((pca_prevalence_sc[:, i-1]+pca_prevalence_sc[:, i]) * 0.5)   # This calculation is because of the life-table format of the model
                lyrs_pca_ns_nodiscount[:, i]  = ((pca_prevalence_ns[:, i-1]+pca_prevalence_ns[:, i]) * 0.5)
                lyrs_pca_nos_nodiscount[:, i]  = ((pca_prevalence_nos[:, i-1]+pca_prevalence_nos[:, i]) * 0.5)
                lyrs_pca_nos_sc_nodiscount[:, i]  = ((pca_prevalence_nos_sc[:, i-1]+pca_prevalence_nos_sc[:, i]) * 0.5)

                # Costs
                if year > 54:
                    costs_tx_screened[:, i] = ((pca_incidence_screened[:, i]
                                               * localised_stage_sc[:, i].T
                                               * tx_costs_local.T).sum(axis=0)

                                              + (pca_incidence_screened[:, i]
                                                * advanced_stage_sc[:, i].T
                                                * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

                    costs_tx_post_screening[:, i] = ((pca_incidence_post_screening[:, i]
                                                     * localised_stage_ns[:, i].T
                                                     * tx_costs_local.T).sum(axis=0)

                                                    + (pca_incidence_post_screening[:, i]
                                                       * advanced_stage_ns[:, i].T
                                                       * tx_costs_adv.T).sum(axis=0)

                                                   * params.relative_cost_clinically_detected[:, i]) # cost of post-screening cancers

                    costs_tx_sc[:, i] = (costs_tx_screened[:, i] + costs_tx_post_screening[:, i]) # total cost in screened arms

                    costs_tx_nos_sc_screened[:, i] = ((pca_incidence_nos_sc_screened[:, i]
                                                      * localised_stage_nos_sc[:, i].T
                                                      * tx_costs_local.T).sum(axis=0)

                                                    + (pca_incidence_nos_sc_screened[:, i]
                                                     * advanced_stage_nos_sc[:, i].T
                                                     * tx_costs_adv.T).sum(axis=0)) # cost of screen-detected cancers

                    costs_tx_nos_sc_post_screening[:, i] = ((pca_incidence_nos_sc_post_screening[:, i]
                                                           * localised_stage_nos[:, i].T
                                                           * tx_costs_local.T).sum(axis=0)

                                                          + (pca_incidence_nos_sc_post_screening[:, i]
                                                           * advanced_stage_nos[:, i].T
                                                           * tx_costs_adv.T).sum(axis=0)

                                                          * params.relative_cost_clinically_detected[:, i]) # cost of post-screening cancers

                    costs_tx_nos_sc[:, i] = costs_tx_nos_sc_screened[:, i] + costs_tx_nos_sc_post_screening[:, i] # total cost in screened arms

                elif year < 55:
                    costs_tx_sc[:, i] = ((pca_incidence_sc[:, i]
                                         * localised_stage_ns[:, i].T
                                         * tx_costs_local.T).sum(axis=0)

                                        + (pca_incidence_sc[:, i]
                                           * advanced_stage_ns[:, i].T
                                           * tx_costs_adv.T).sum(axis=0)

                                       * params.relative_cost_clinically_detected[:, i])

                    costs_tx_nos_sc[:, i] = ((pca_incidence_nos_sc[:, i]
                                             * localised_stage_nos[:, i].T
                                             * tx_costs_local.T).sum(axis=0)

                                            + (pca_incidence_nos_sc[:, i]
                                             * advanced_stage_nos[:, i].T
                                             * tx_costs_adv.T).sum(axis=0)

                                           * params.relative_cost_clinically_detected[:, i])

                costs_tx_ns[:, i] = ((pca_incidence_ns[:, i]
                                     * localised_stage_ns[:, i].T
                                     * tx_costs_local.T).sum(axis=0)

                                    + (pca_incidence_ns[:, i]
                                     * advanced_stage_ns[:, i].T
                                     * tx_costs_adv.T).sum(axis=0)

                                   * params.relative_cost_clinically_detected[:, i])

                costs_tx_nos[:, i] = ((pca_incidence_nos[:, i]
                                      * localised_stage_nos[:, i].T
                                      * tx_costs_local.T).sum(axis=0)

                                     + (pca_incidence_nos[:, i]
                                      * advanced_stage_nos[:, i].T
                                      * tx_costs_adv.T).sum(axis=0)

                                    * params.relative_cost_clinically_detected[:, i])

            ############
            # Outcomes #
            ############

            # Incident cases (total)
            cases_prs = pca_incidence_sc + pca_incidence_ns + pca_incidence_nos + pca_incidence_nos_sc

            # PCa alive
            pca_alive_prs = pca_alive_sc + pca_alive_ns + pca_alive_nos + pca_alive_nos_sc

            # Healthy
            healthy_prs = healthy_sc + healthy_ns + healthy_nos + healthy_nos_sc

            # Overdiagnosed cases
            if od_by_risk==False:
                # Baseline analyses
                overdiagnosis_sc = pca_incidence_screened * params.p_overdiagnosis_psa.T[:, year-45:]
                overdiagnosis_nos_sc = pca_incidence_nos_sc_screened * params.p_overdiagnosis_psa.T[:, year-45:]

            elif od_by_risk==True:
                # Sensitivity analyses - overdiagnosis varying inversely by polygenic risk doi:10.1038/gim.2014.192
                overdiagnosis_sc = pca_incidence_screened * params.p_overdiagnosis_psa.T[:, year-45:] * (1/a_risk.loc[year, 'rr_high'])
                overdiagnosis_nos_sc = pca_incidence_nos_sc_screened * params.p_overdiagnosis_psa.T[:, year-45:] * (1/a_risk.loc[year, 'rr_low'])

            overdiagnosis_prs = overdiagnosis_sc + overdiagnosis_nos_sc

            # Deaths from other causes (screened armss)
            deaths_sc_other_prs = pca_death_other_sc + healthy_death_other_sc
            deaths_nos_sc_other_prs = pca_death_other_nos_sc + healthy_death_other_nos_sc

            # Deaths from other causes (non-screened arms)
            deaths_ns_other_prs = pca_death_other_ns + healthy_death_other_ns
            deaths_nos_other_prs = pca_death_other_nos + healthy_death_other_nos

            # Total deaths from other causes
            ################################
            deaths_other_prs = (deaths_sc_other_prs
                               + deaths_ns_other_prs
                               + deaths_nos_other_prs
                               + deaths_nos_sc_other_prs)

            # Deaths from prosate cancer (screened arms)
            deaths_pca_sc_nos_sc_prs = pca_death_sc + pca_death_nos_sc

            # Deaths from prosate cancer (non-screened arms)
            deaths_pca_ns_nos_prs = pca_death_ns + pca_death_nos

            # Deaths from prosate cancer (total)
            ####################################
            deaths_pca_prs = (pca_death_sc
                             + pca_death_ns
                             + pca_death_nos
                             + pca_death_nos_sc)

            ##############
            # Life-years #
            ##############

            # Healthy life-years (screened arm)
            lyrs_healthy_sc_nodiscount_prs = healthy_sc - (0.5 * (healthy_death_other_sc+pca_incidence_sc))
            lyrs_healthy_sc_discount_prs = lyrs_healthy_sc_nodiscount_prs * params.discount_factor[:total_cycles]

            lyrs_healthy_nos_sc_nodiscount_prs = healthy_nos_sc - (0.5 * (healthy_death_other_nos_sc+pca_incidence_nos_sc))
            lyrs_healthy_nos_sc_discount_prs = lyrs_healthy_nos_sc_nodiscount_prs * params.discount_factor[:total_cycles]

            # Healthy life-years (non-screened arm)
            lyrs_healthy_ns_nodiscount_prs = healthy_ns - (0.5 * (healthy_death_other_ns+pca_incidence_ns))
            lyrs_healthy_ns_discount_prs = lyrs_healthy_ns_nodiscount_prs * params.discount_factor[:total_cycles]

            lyrs_healthy_nos_nodiscount_prs = healthy_nos - (0.5 * (healthy_death_other_nos+pca_incidence_nos))
            lyrs_healthy_nos_discount_prs = lyrs_healthy_nos_nodiscount_prs * params.discount_factor[:total_cycles]

            # Total healthy life-years
            lyrs_healthy_nodiscount_prs = (lyrs_healthy_sc_nodiscount_prs
                                           + lyrs_healthy_ns_nodiscount_prs
                                           + lyrs_healthy_nos_nodiscount_prs
                                           + lyrs_healthy_nos_sc_nodiscount_prs)

            lyrs_healthy_discount_prs = (lyrs_healthy_sc_discount_prs
                                         + lyrs_healthy_ns_discount_prs
                                         + lyrs_healthy_nos_discount_prs
                                         + lyrs_healthy_nos_sc_discount_prs)

            # Life-years with prostate cancer in screened arms
            lyrs_pca_sc_discount = lyrs_pca_sc_nodiscount * params.discount_factor[:total_cycles]
            lyrs_pca_nos_sc_discount = lyrs_pca_nos_sc_nodiscount * params.discount_factor[:total_cycles]

            # Life-years with prostate cancer in non-screened arms
            lyrs_pca_ns_discount = lyrs_pca_ns_nodiscount * params.discount_factor[:total_cycles]
            lyrs_pca_nos_discount = lyrs_pca_nos_nodiscount * params.discount_factor[:total_cycles]

            #  Life-years with prostate cancer in both arms
            lyrs_pca_nodiscount_prs = (lyrs_pca_sc_nodiscount
                                       + lyrs_pca_ns_nodiscount
                                       + lyrs_pca_nos_nodiscount
                                       + lyrs_pca_nos_sc_nodiscount)

            lyrs_pca_discount_prs = (lyrs_pca_sc_discount
                                     + lyrs_pca_ns_discount
                                     + lyrs_pca_nos_discount
                                     + lyrs_pca_nos_sc_discount)

            # Total Life-years
            ##################
            lyrs_nodiscount_prs = lyrs_healthy_nodiscount_prs + lyrs_pca_nodiscount_prs
            lyrs_discount_prs = lyrs_healthy_discount_prs + lyrs_pca_discount_prs

            #########
            # QALYs #
            #########

            # QALYs (healthy life) - screened arms
            qalys_healthy_sc_nodiscount_prs = lyrs_healthy_sc_nodiscount_prs * params.utility_background[:, year-45:]
            qalys_healthy_sc_discount_prs = lyrs_healthy_sc_discount_prs * params.utility_background[:, year-45:]
            qalys_healthy_nos_sc_nodiscount_prs = lyrs_healthy_nos_sc_nodiscount_prs * params.utility_background[:, year-45:]
            qalys_healthy_nos_sc_discount_prs = lyrs_healthy_nos_sc_discount_prs * params.utility_background[:, year-45:]

            # QALYs (healthy life) - non-screened arms
            qalys_healthy_ns_nodiscount_prs = lyrs_healthy_ns_nodiscount_prs * params.utility_background[:, year-45:]
            qalys_healthy_ns_discount_prs = lyrs_healthy_ns_discount_prs * params.utility_background[:, year-45:]
            qalys_healthy_nos_nodiscount_prs = lyrs_healthy_nos_nodiscount_prs * params.utility_background[:, year-45:]
            qalys_healthy_nos_discount_prs = lyrs_healthy_nos_discount_prs * params.utility_background[:, year-45:]

            # Total QALYs (healthy life)
            qalys_healthy_nodiscount_prs = lyrs_healthy_nodiscount_prs * params.utility_background[:, year-45:]
            qalys_healthy_discount_prs = lyrs_healthy_discount_prs * params.utility_background[:, year-45:]

            # QALYS with prostate cancer - screened arms
            qalys_pca_sc_nodiscount_prs = lyrs_pca_sc_nodiscount * params.utility_pca[:, year-45:]
            qalys_pca_sc_discount_prs = lyrs_pca_sc_discount * params.utility_pca[:, year-45:]
            qalys_pca_nos_sc_nodiscount_prs = lyrs_pca_nos_sc_nodiscount * params.utility_pca[:, year-45:]
            qalys_pca_nos_sc_discount_prs = lyrs_pca_nos_sc_discount * params.utility_pca[:, year-45:]

            # QALYS with prostate cancer - non-screened arms
            qalys_pca_ns_nodiscount_prs = lyrs_pca_ns_nodiscount * params.utility_pca[:, year-45:]
            qalys_pca_ns_discount_prs = lyrs_pca_ns_discount * params.utility_pca[:, year-45:]
            qalys_pca_nos_nodiscount_prs = lyrs_pca_nos_nodiscount * params.utility_pca[:, year-45:]
            qalys_pca_nos_discount_prs = lyrs_pca_nos_discount * params.utility_pca[:, year-45:]

            # Total QALYS with prostate cancer
            qalys_pca_nodiscount_prs = lyrs_pca_nodiscount_prs * params.utility_pca[:, year-45:]
            qalys_pca_discount_prs = lyrs_pca_discount_prs * params.utility_pca[:, year-45:]

            # Total QALYs
            #############
            qalys_nodiscount_prs = qalys_healthy_nodiscount_prs + qalys_pca_nodiscount_prs
            qalys_discount_prs = qalys_healthy_discount_prs + qalys_pca_discount_prs

            # Costs of risk-stratification
            cost_screening_prs = params.cost_prs * params.uptake_prs * params.pop.loc[year, :].values # There is no discounting of risk-stratification as done at year 1 of the model.

            #############
            # PSA tests #
            #############

            # Costs of PSA testing in non-screened arms
            n_psa_tests_ns_prs = (pca_incidence_ns / params.p_biopsy_ns[:, year-45:]) * params.n_psa_tests[:, year-45:]

            cost_psa_testing_ns_nodiscount_prs = (n_psa_tests_ns_prs
                                                  * params.cost_psa[:, year-45:]
                                                  * params.relative_cost_clinically_detected[:, year-45:])

            cost_psa_testing_ns_discount_prs = cost_psa_testing_ns_nodiscount_prs * params.discount_factor[:total_cycles]

            n_psa_tests_nos_prs = (pca_incidence_nos / params.p_biopsy_ns[:, year-45:]) * params.n_psa_tests[:, year-45:]

            cost_psa_testing_nos_nodiscount_prs = (n_psa_tests_nos_prs
                                                   * params.cost_psa[:, year-45:]
                                                   * params.relative_cost_clinically_detected[:, year-45:])

            cost_psa_testing_nos_discount_prs = cost_psa_testing_nos_nodiscount_prs * params.discount_factor[:total_cycles]

            # Costs of PSA testing in screened arms
            if year > 54:
                # Get the screened years
                lyrs_healthy_screened_nodiscount_prs = np.array([np.zeros(length_df)] * params.sims)
                lyrs_healthy_screened_nodiscount_prs[:,:length_screen] = lyrs_healthy_sc_nodiscount_prs[:,:length_screen].copy()
                lyrs_healthy_screened_nodiscount_prs[:,length_screen:] = 0

                # Population-level PSA testing during screening phase
                n_psa_tests_screened_prs = lyrs_healthy_screened_nodiscount_prs * params.uptake_psa / 4

                # Assuming all cancers are clinically detected in the post-screening phase
                n_psa_tests_post_screening_prs = ((pca_incidence_post_screening / params.p_biopsy_ns[:, year-45:])
                                                  * params.n_psa_tests[:, year-45:])

                n_psa_tests_sc_prs = n_psa_tests_screened_prs + n_psa_tests_post_screening_prs

                cost_psa_testing_sc_nodiscount_prs = ((n_psa_tests_screened_prs * params.cost_psa[:, year-45:])
                                                      + (n_psa_tests_post_screening_prs
                                                         * params.cost_psa[:, year-45:]
                                                         * params.relative_cost_clinically_detected[:, year-45:]))

                # PSA tests in the not offered screening but screened anyway group
                # Get the screened years
                lyrs_healthy_nos_sc_screened_nodiscount_prs = np.array([np.zeros(length_df)] * params.sims)
                lyrs_healthy_nos_sc_screened_nodiscount_prs[:,:length_screen] = lyrs_healthy_nos_sc_nodiscount_prs[:,:length_screen].copy()
                lyrs_healthy_nos_sc_screened_nodiscount_prs[:,length_screen:] = 0

                # Population-level PSA testing during screening phase
                n_psa_tests_nos_sc_screened_prs = lyrs_healthy_nos_sc_screened_nodiscount_prs * params.uptake_psa / 4

                # Assuming all cancers are clinically detected in the post-screening phase
                n_psa_tests_nos_sc_post_screening_prs = ((pca_incidence_nos_sc_post_screening / params.p_biopsy_ns[:, year-45:])
                                                         * params.n_psa_tests[:, year-45:])

                n_psa_tests_nos_sc_prs = n_psa_tests_nos_sc_screened_prs + n_psa_tests_nos_sc_post_screening_prs

                cost_psa_testing_nos_sc_nodiscount_prs = ((n_psa_tests_nos_sc_screened_prs * params.cost_psa[:, year-45:])
                                                          + (n_psa_tests_nos_sc_post_screening_prs
                                                             * params.cost_psa[:, year-45:]
                                                             * params.relative_cost_clinically_detected[:, year-45:]))

            elif year < 55:
                lyrs_healthy_screened_nodiscount_prs = np.array([np.zeros(length_df)] * params.sims)
                n_psa_tests_sc_prs = (pca_incidence_sc / params.p_biopsy_ns[:, year-45:]) * params.n_psa_tests[:, year-45:]
                n_psa_tests_nos_sc_prs = (pca_incidence_nos_sc / params.p_biopsy_ns[:, year-45:]) * params.n_psa_tests[:, year-45:]

                cost_psa_testing_sc_nodiscount_prs = (n_psa_tests_sc_prs
                                                      * params.cost_psa[:, year-45:]
                                                      * params.relative_cost_clinically_detected[:, year-45:])

                cost_psa_testing_nos_sc_nodiscount_prs = (n_psa_tests_nos_sc_prs
                                                          * params.cost_psa[:, year-45:]
                                                          * params.relative_cost_clinically_detected[:, year-45:])

            cost_psa_testing_sc_discount_prs = cost_psa_testing_sc_nodiscount_prs * params.discount_factor[:total_cycles]
            cost_psa_testing_nos_sc_discount_prs = cost_psa_testing_nos_sc_nodiscount_prs * params.discount_factor[:total_cycles]

            # Total costs of PSA testing
            ############################
            n_psa_tests_prs = (n_psa_tests_sc_prs
                               + n_psa_tests_ns_prs
                               + n_psa_tests_nos_prs
                               + n_psa_tests_nos_sc_prs)

            cost_psa_testing_nodiscount_prs = (cost_psa_testing_sc_nodiscount_prs
                                               + cost_psa_testing_ns_nodiscount_prs
                                               + cost_psa_testing_nos_nodiscount_prs
                                               + cost_psa_testing_nos_sc_nodiscount_prs)

            cost_psa_testing_discount_prs = (cost_psa_testing_sc_discount_prs
                                             + cost_psa_testing_ns_discount_prs
                                             + cost_psa_testing_nos_discount_prs
                                             + cost_psa_testing_nos_sc_discount_prs)

            #######
            # MRI #
            #######

            # Costs of MRI - screened arms
            if year < 55:
                n_mri_sc_prs = pca_incidence_sc / params.p_biopsy_ns[:, year-45:]
                cost_mri_sc_nodiscount_prs = (n_mri_sc_prs
                                              * np.array([params.cost_mri]).T
                                              * params.relative_cost_clinically_detected[:, year-45:])

                n_mri_nos_sc_prs = pca_incidence_nos_sc / params.p_biopsy_ns[:, year-45:]
                cost_mri_nos_sc_nodiscount_prs = (n_mri_nos_sc_prs
                                                  * np.array([params.cost_mri]).T
                                                  * params.relative_cost_clinically_detected[:, year-45:])


            elif year > 54:
                # Screen-detected cancers
                n_mri_screened_prs = (n_psa_tests_screened_prs * params.p_mri[:, year-45:]) * params.uptake_mri
                cost_mri_screened_nodiscount_prs = n_mri_screened_prs * np.array([params.cost_mri]).T

                # Assuming all cancers are clinically detected in the post-screening phase
                n_mri_post_screening_prs = pca_incidence_post_screening / params.p_biopsy_ns[:, year-45:]
                cost_mri_post_screening_nodiscount_prs = (n_mri_post_screening_prs
                                                          * np.array([params.cost_mri]).T
                                                          * params.relative_cost_clinically_detected[:, year-45:])

                n_mri_sc_prs = n_mri_screened_prs + n_mri_post_screening_prs

                # Total cost of MRI in screened arms
                cost_mri_sc_nodiscount_prs = cost_mri_screened_nodiscount_prs + cost_mri_post_screening_nodiscount_prs

                # MRI in those who were not above risk threshold but screened anyway
                n_mri_nos_sc_screened_prs = n_psa_tests_nos_sc_prs * params.p_mri[:, year-45:]
                cost_mri_nos_sc_screened_nodiscount_prs = n_mri_nos_sc_screened_prs * np.array([params.cost_mri]).T

                # Assuming all cancers are clinically detected in the post-screening phase
                n_mri_nos_sc_post_screening_prs = pca_incidence_nos_sc_post_screening / params.p_biopsy_ns[:, year-45:]
                cost_mri_nos_sc_post_screening_nodiscount_prs = n_mri_nos_sc_post_screening_prs * np.array([params.cost_mri]).T

                # Total MRI
                n_mri_nos_sc_prs = n_mri_nos_sc_screened_prs + n_mri_nos_sc_post_screening_prs
                cost_mri_nos_sc_nodiscount_prs = (cost_mri_nos_sc_screened_nodiscount_prs
                                                  + cost_mri_nos_sc_post_screening_nodiscount_prs)

            cost_mri_sc_discount_prs = cost_mri_sc_nodiscount_prs * params.discount_factor[:total_cycles]
            cost_mri_nos_sc_discount_prs = cost_mri_nos_sc_nodiscount_prs * params.discount_factor[:total_cycles]

            # Costs of MRI - non-screened arms
            n_mri_ns_prs = pca_incidence_ns / params.p_biopsy_ns[:, year-45:]
            cost_mri_ns_nodiscount_prs = (n_mri_ns_prs
                                          * np.array([params.cost_mri]).T
                                          * params.relative_cost_clinically_detected[:, year-45:])

            cost_mri_ns_discount_prs = cost_mri_ns_nodiscount_prs * params.discount_factor[:total_cycles]

            n_mri_nos_prs = pca_incidence_nos / params.p_biopsy_ns[:, year-45:]
            cost_mri_nos_nodiscount_prs = ((n_mri_nos_prs * np.array([params.cost_mri]).T)
                                           * params.relative_cost_clinically_detected[:, year-45:])

            cost_mri_nos_discount_prs = cost_mri_nos_nodiscount_prs * params.discount_factor[:total_cycles]

            # Total costs of mri
            #######################
            n_mri_prs = (n_mri_sc_prs
                         + n_mri_ns_prs
                         + n_mri_nos_prs
                         + n_mri_nos_sc_prs)

            cost_mri_nodiscount_prs = (cost_mri_sc_nodiscount_prs
                                       + cost_mri_ns_nodiscount_prs
                                       + cost_mri_nos_nodiscount_prs
                                       + cost_mri_nos_sc_nodiscount_prs)

            cost_mri_discount_prs = (cost_mri_sc_discount_prs
                                     + cost_mri_ns_discount_prs
                                     + cost_mri_nos_discount_prs
                                     + cost_mri_nos_sc_discount_prs)

            ##########
            # Biopsy #
            ##########

            # Costs of biopsy - screened arms
            if year > 54:
                # Screen-detected cancers
                n_biopsies_screened_prs = n_mri_screened_prs * params.n_biopsies_primari[:, year-45:]
                cost_biopsy_screened_nodiscount_prs = n_biopsies_screened_prs * params.cost_biopsy[:, year-45:]

                # Assuming all cancers are clinically detected in the post-screening phase
                n_biopsies_post_screening_prs = n_mri_post_screening_prs * params.n_biopsies_primari[:, year-45:]

                cost_biopsies_post_screening_nodiscount_prs = (n_biopsies_post_screening_prs
                                                               * params.cost_biopsy[:, year-45:]
                                                               * params.relative_cost_clinically_detected[:, year-45:])

                # Total biopsies screened arm
                n_biopsies_sc_prs = n_biopsies_screened_prs + n_biopsies_post_screening_prs
                cost_biopsy_sc_nodiscount_prs = cost_biopsy_screened_nodiscount_prs + cost_biopsies_post_screening_nodiscount_prs

                n_biopsies_nos_sc_screened_prs = n_mri_nos_sc_post_screening_prs * params.n_biopsies_primari[:, year-45:]
                cost_biopsy_nos_sc_screened_nodiscount_prs = (n_biopsies_nos_sc_screened_prs * params.cost_biopsy[:, year-45:])

                # Assuming all cancers are clinically detected in the post-screening phase
                n_biopsies_nos_sc_post_screening_prs = n_mri_nos_sc_post_screening_prs * params.n_biopsies_primari[:, year-45:]

                cost_biopsies_nos_sc_post_screening_nodiscount_prs = (n_biopsies_nos_sc_post_screening_prs
                                                                      * params.cost_biopsy[:, year-45:]
                                                                      * params.relative_cost_clinically_detected[:, year-45:])

                # Total biopsies in non-eligible, but screened anyway arm
                n_biopsies_nos_sc_prs = n_biopsies_nos_sc_screened_prs + n_biopsies_nos_sc_post_screening_prs

                # Total cost of biopsies
                cost_biopsy_nos_sc_nodiscount_prs = (cost_biopsy_nos_sc_screened_nodiscount_prs
                                                     + cost_biopsies_nos_sc_post_screening_nodiscount_prs)

            elif year < 55:
                n_biopsies_sc_prs = n_mri_sc_prs * params.n_biopsies_primari[:, year-45:]

                cost_biopsy_sc_nodiscount_prs = (n_biopsies_sc_prs
                                                 * params.cost_biopsy[:, year-45:]
                                                 * params.relative_cost_clinically_detected[:, year-45:])

                n_biopsies_nos_sc_prs = n_mri_nos_sc_prs * params.n_biopsies_primari[:, year-45:]

                cost_biopsy_nos_sc_nodiscount_prs = (n_biopsies_nos_sc_prs
                                                     * params.cost_biopsy[:, year-45:]
                                                     * params.relative_cost_clinically_detected[:, year-45:])

            cost_biopsy_sc_discount_prs = cost_biopsy_sc_nodiscount_prs * params.discount_factor[:total_cycles]
            cost_biopsy_nos_sc_discount_prs = cost_biopsy_nos_sc_nodiscount_prs * params.discount_factor[:total_cycles]

            # Costs of biopsy - non-screened arms
            n_biopsies_ns_prs = n_mri_ns_prs * params.n_biopsies_primari[:, year-45:]

            cost_biopsy_ns_nodiscount_prs = (n_biopsies_ns_prs
                                             * params.cost_biopsy[:, year-45:]
                                             * params.relative_cost_clinically_detected[:, year-45:])

            cost_biopsy_ns_discount_prs = cost_biopsy_ns_nodiscount_prs * params.discount_factor[:total_cycles]

            n_biopsies_nos_prs = n_mri_nos_prs * params.n_biopsies_primari[:, year-45:]
            cost_biopsy_nos_nodiscount_prs = (n_biopsies_nos_prs
                                              * params.cost_biopsy[:, year-45:]
                                              * params.relative_cost_clinically_detected[:, year-45:])

            cost_biopsy_nos_discount_prs = cost_biopsy_nos_nodiscount_prs * params.discount_factor[:total_cycles]

            # Total costs of biopsy
            #######################
            n_biopsies_prs = (n_biopsies_sc_prs
                              + n_biopsies_ns_prs
                              + n_biopsies_nos_prs
                              + n_biopsies_nos_sc_prs)

            cost_biopsy_nodiscount_prs = (cost_biopsy_sc_nodiscount_prs
                                          + cost_biopsy_ns_nodiscount_prs
                                          + cost_biopsy_nos_nodiscount_prs
                                          + cost_biopsy_nos_sc_nodiscount_prs)

            cost_biopsy_discount_prs = (cost_biopsy_sc_discount_prs
                                        + cost_biopsy_ns_discount_prs
                                        + cost_biopsy_nos_discount_prs
                                        + cost_biopsy_nos_sc_discount_prs)

            # Cost of staging in the screened arms
            if year > 54:
                cost_staging_screened_nodiscount_prs = (params.cost_assessment
                                                        * advanced_stage_sc.T
                                                        * pca_incidence_screened.T).T

                cost_staging_post_screening_nodiscount_prs = (params.cost_assessment
                                                              * advanced_stage_ns.T
                                                              * pca_incidence_post_screening.T
                                                              * params.relative_cost_clinically_detected[:, year-45:].T).T

                cost_staging_sc_nodiscount_prs = (cost_staging_screened_nodiscount_prs
                                                  + cost_staging_post_screening_nodiscount_prs)

                cost_staging_nos_sc_screened_nodiscount_prs = (params.cost_assessment
                                                               * advanced_stage_nos_sc.T
                                                               * pca_incidence_nos_sc_screened.T).T

                cost_staging_nos_sc_post_screening_nodiscount_prs = (params.cost_assessment
                                                                     * advanced_stage_nos.T
                                                                     * pca_incidence_nos_sc_post_screening.T
                                                                     * params.relative_cost_clinically_detected[:, year-45:].T).T

                cost_staging_nos_sc_nodiscount_prs = (cost_staging_nos_sc_screened_nodiscount_prs
                                                     + cost_staging_nos_sc_post_screening_nodiscount_prs)

            if year < 55:
                cost_staging_sc_nodiscount_prs = (params.cost_assessment
                                                  * advanced_stage_ns.T
                                                  * pca_incidence_sc.T
                                                  * params.relative_cost_clinically_detected[:, year-45:].T).T

                cost_staging_nos_sc_nodiscount_prs = (params.cost_assessment
                                                      * advanced_stage_nos.T
                                                      * pca_incidence_nos_sc.T
                                                      * params.relative_cost_clinically_detected[:, year-45:].T).T

            cost_staging_sc_discount_prs = cost_staging_sc_nodiscount_prs * params.discount_factor[:total_cycles]
            cost_staging_nos_sc_discount_prs = cost_staging_nos_sc_nodiscount_prs * params.discount_factor[:total_cycles]

            # Cost of staging in the non-screened arms
            cost_staging_ns_nodiscount_prs = (params.cost_assessment
                                              * advanced_stage_ns.T
                                              * pca_incidence_ns.T
                                              * params.relative_cost_clinically_detected[:, year-45:].T).T

            cost_staging_ns_discount_prs = cost_staging_ns_nodiscount_prs * params.discount_factor[:total_cycles]

            cost_staging_nos_nodiscount_prs = (params.cost_assessment
                                               * advanced_stage_nos.T
                                               * pca_incidence_nos.T
                                               * params.relative_cost_clinically_detected[:, year-45:].T).T

            cost_staging_nos_discount_prs = cost_staging_nos_nodiscount_prs * params.discount_factor[:total_cycles]

            # Total costs of staging
            ########################
            cost_staging_nodiscount_prs = (cost_staging_sc_nodiscount_prs
                                           + cost_staging_ns_nodiscount_prs
                                           + cost_staging_nos_nodiscount_prs
                                           + cost_staging_nos_sc_nodiscount_prs)

            cost_staging_discount_prs = (cost_staging_sc_discount_prs
                                         + cost_staging_ns_discount_prs
                                         + cost_staging_nos_discount_prs
                                         + cost_staging_nos_sc_discount_prs)

            # Cost of treatment in screened arms
            cost_tx_sc_nodiscount_prs = costs_tx_sc * params.discount_factor[:total_cycles]
            cost_tx_nos_sc_nodiscount_prs = costs_tx_nos_sc * params.discount_factor[:total_cycles]

            # Cost of treatment in non-screened arms
            cost_tx_ns_nodiscount_prs = costs_tx_ns * params.discount_factor[:total_cycles]
            cost_tx_nos_nodiscount_prs = costs_tx_nos * params.discount_factor[:total_cycles]

            # Total costs of treatment
            ##########################
            cost_tx_nodiscount_prs = (costs_tx_sc
                                      + costs_tx_ns
                                      + costs_tx_nos
                                      + costs_tx_nos_sc)

            cost_tx_discount_prs = cost_tx_nodiscount_prs * params.discount_factor[:total_cycles]

            # Costs of palliation and death in screened arm
            cost_eol_sc_nodiscount_prs = (params.cost_pca_death * pca_death_sc.T).T
            cost_eol_sc_discount_prs = cost_eol_sc_nodiscount_prs * params.discount_factor[:total_cycles]
            cost_eol_nos_sc_nodiscount_prs = (params.cost_pca_death * pca_death_nos_sc.T).T
            cost_eol_nos_sc_discount_prs = cost_eol_nos_sc_nodiscount_prs * params.discount_factor[:total_cycles]

            # Costs of palliation and death in non-screened arm
            cost_eol_ns_nodiscount_prs = (params.cost_pca_death * pca_death_ns.T).T
            cost_eol_ns_discount_prs = cost_eol_ns_nodiscount_prs * params.discount_factor[:total_cycles]
            cost_eol_nos_nodiscount_prs = (params.cost_pca_death * pca_death_nos.T).T
            cost_eol_nos_discount_prs = cost_eol_nos_nodiscount_prs * params.discount_factor[:total_cycles]

            # Total costs of palliation and death
            cost_eol_nodiscount_prs = (cost_eol_sc_nodiscount_prs
                                       + cost_eol_ns_nodiscount_prs
                                       + cost_eol_nos_nodiscount_prs
                                       + cost_eol_nos_sc_nodiscount_prs)

            cost_eol_discount_prs = (cost_eol_sc_discount_prs
                                     + cost_eol_ns_discount_prs
                                     + cost_eol_nos_discount_prs
                                     + cost_eol_nos_sc_discount_prs)

            # TOTAL COSTS PRS-BASED SCREENING
            #################################
            cost_nodiscount_prs = (cost_psa_testing_nodiscount_prs
                                   + cost_mri_nodiscount_prs
                                   + cost_biopsy_nodiscount_prs
                                   + cost_staging_nodiscount_prs
                                   + cost_tx_nodiscount_prs
                                   + cost_eol_nodiscount_prs)

            cost_discount_prs = (cost_psa_testing_discount_prs
                                 + cost_mri_discount_prs
                                 + cost_biopsy_discount_prs
                                 + cost_staging_discount_prs
                                 + cost_tx_discount_prs
                                 + cost_eol_discount_prs)

            # Generate a mean dataframe for each age cohort
            cohort = pd.DataFrame({
                'age': age,
                'pca_cases': mean(cases_prs),
                'pca_cases_sc': mean(pca_incidence_sc),
                'pca_cases_ns': mean(pca_incidence_ns),
                'pca_cases_nos': mean(pca_incidence_nos),
                'pca_cases_nos_sc': mean(pca_incidence_nos_sc),
                'screen-detected cases': mean((pca_incidence_screened+pca_incidence_nos_sc_screened)),
                'screen-detected cases_sc': mean(pca_incidence_screened),
                'screen-detected cases_nos_sc': mean(pca_incidence_nos_sc_screened),
                'post-screening cases': mean((pca_incidence_post_screening+pca_incidence_nos_sc_post_screening)),
                'post-screening cases_sc': mean(pca_incidence_post_screening),
                'post-screening cases_nos_sc': mean(pca_incidence_nos_sc_post_screening),
                'pca_misclassified': mean((pca_misclassified+pca_misclassified_nos_sc)),
                'pca_misclassified_sc': mean(pca_misclassified),
                'pca_misclassified_nos_sc': mean(pca_misclassified_nos_sc),
                'overdiagnosis': mean(overdiagnosis_prs),
                'overdiagnosis_sc': mean(overdiagnosis_sc),
                'overdiagnosis_nos_sc': mean(overdiagnosis_nos_sc),
                'deaths_other': mean(deaths_other_prs),
                'deaths_other_sc': mean(deaths_sc_other_prs),
                'deaths_other_ns': mean(deaths_ns_other_prs),
                'deaths_other_nos': mean(deaths_nos_other_prs),
                'deaths_other_nos_sc': mean(deaths_nos_sc_other_prs),
                'deaths_pca': mean(deaths_pca_prs),
                'deaths_pca_sc': mean(pca_death_sc),
                'deaths_pca_ns': mean(pca_death_ns),
                'deaths_pca_nos': mean(pca_death_nos),
                'deaths_pca_nos_sc': mean(pca_death_nos_sc),
                'pca_alive': mean(pca_alive_prs),
                'pca_alive_sc': mean(pca_alive_sc),
                'pca_alive_ns': mean(pca_alive_ns),
                'pca_alive_nos': mean(pca_alive_nos),
                'pca_alive_nos_sc': mean(pca_alive_nos_sc),
                'healthy': mean(healthy_prs),
                'healthy_sc': mean(healthy_sc),
                'healthy_ns': mean(healthy_ns),
                'healthy_nos': mean(healthy_nos),
                'healthy_nos_sc': mean(healthy_nos_sc),
                'psa_tests': mean(n_psa_tests_prs),
                'n_mri': mean(n_mri_prs),
                'n_biopsies': mean(n_biopsies_prs),
                'lyrs_healthy_nodiscount': mean(lyrs_healthy_nodiscount_prs),
                'lyrs_healthy_discount': mean(lyrs_healthy_discount_prs),
                'lyrs_pca_discount': mean(lyrs_pca_discount_prs),
                'total_lyrs_discount': mean(lyrs_discount_prs),
                'qalys_healthy_discount': mean(qalys_healthy_discount_prs),
                'qalys_pca_discount': mean(qalys_pca_discount_prs),
                'total_qalys_discount': mean(qalys_discount_prs),
                'cost_psa_testing_discount': mean(cost_psa_testing_discount_prs),
                'cost_biopsy_discount': mean(cost_biopsy_discount_prs),
                'cost_mri_discount': mean(cost_mri_discount_prs),
                'cost_staging_discount': mean(cost_staging_discount_prs),
                'cost_treatment_discount': mean(cost_tx_discount_prs),
                'costs_eol_discount': mean(cost_eol_discount_prs),
                'total_cost_discount': mean(cost_discount_prs)
            })

            # Totals for each age cohort
            outcomes = pd.DataFrame({
                'cohort_prs_at_start': [year],
                'pca_cases': [total(cases_prs)],
                'pca_cases_sc': [total(pca_incidence_sc)],
                'pca_cases_ns': [total(pca_incidence_ns)],
                'pca_cases_nos': [total(pca_incidence_nos)],
                'pca_cases_nos_sc': [total(pca_incidence_nos_sc)],
                'screen-detected cases': [total(pca_incidence_screened+pca_incidence_nos_sc_screened)],
                'screen-detected cases_sc': [total(pca_incidence_screened)],
                'screen-detected cases_nos_sc': [total(pca_incidence_nos_sc_screened)],
                'post-screening cases': [total(pca_incidence_post_screening+pca_incidence_nos_sc_post_screening)],
                'post-screening cases_sc': [total(pca_incidence_post_screening)],
                'post-screening cases_nos_sc': [total(pca_incidence_nos_sc_post_screening)],
                'pca_misclassified': [total((pca_misclassified+pca_misclassified_nos_sc))],
                'pca_misclassified_sc': [total(pca_misclassified)],
                'pca_misclassified_nos_sc': [total(pca_misclassified_nos_sc)],
                'overdiagnosis': [total(overdiagnosis_prs)],
                'overdiagnosis_sc': [total(overdiagnosis_sc)],
                'overdiagnosis_nos_sc': [total(overdiagnosis_nos_sc)],
                'pca_deaths': [total(deaths_pca_prs)],
                'pca_deaths_sc': [total(pca_death_sc)],
                'pca_deaths_ns': [total(pca_death_ns)],
                'pca_deaths_nos': [total(pca_death_nos)],
                'pca_deaths_nos_sc': [total(pca_death_nos_sc)],
                'deaths_other_causes': [total(deaths_other_prs)],
                'deaths_other_sc': [total(deaths_sc_other_prs)],
                'deaths_other_ns': [total(deaths_ns_other_prs)],
                'deaths_other_nos': [total(deaths_nos_other_prs)],
                'deaths_other_nos_sc': [total(deaths_nos_sc_other_prs)],
                'lyrs_healthy_discounted': [total(lyrs_healthy_discount_prs)],
                'lyrs_pca_discounted': [total(lyrs_pca_discount_prs)],
                'lyrs_undiscounted': [total(lyrs_nodiscount_prs)],
                'lyrs_discounted': [total(lyrs_discount_prs)],
                'qalys_healthy_discounted': [total(qalys_healthy_discount_prs)],
                'qalys_pca_discounted': [total(qalys_pca_discount_prs)],
                'qalys_undiscounted': [total(qalys_nodiscount_prs)],
                'qalys_discounted': [total(qalys_discount_prs)],
                'cost_screening': [np.mean(cost_screening_prs)],
                'cost_psa_testing_undiscounted': [total(cost_psa_testing_nodiscount_prs)],
                'cost_psa_testing_discounted': [total(cost_psa_testing_discount_prs)],
                'cost_mri_undiscounted': [total(cost_mri_nodiscount_prs)],
                'cost_mri_discounted': [total(cost_mri_discount_prs)],
                'cost_biopsy_undiscounted': [total(cost_biopsy_nodiscount_prs)],
                'cost_biopsy_discounted': [total(cost_biopsy_discount_prs)],
                'cost_staging_undiscounted': [total(cost_staging_nodiscount_prs)],
                'cost_staging_discounted': [total(cost_staging_discount_prs)],
                'cost_treatment_undiscounted': [total(cost_tx_nodiscount_prs)],
                'cost_treatment_discounted': [total(cost_tx_discount_prs)],
                'cost_eol_undiscounted': [total(cost_eol_nodiscount_prs)],
                'cost_eol_discounted': [total(cost_eol_discount_prs)],
                'costs_undiscounted': [total(cost_nodiscount_prs)+np.mean(cost_screening_prs)],
                'costs_discounted': [total(cost_discount_prs)+np.mean(cost_screening_prs)],
                'n_psa_tests': [total(n_psa_tests_prs)],
                'n_mri': [total(n_mri_prs)],
                'n_biopsies': [total(n_biopsies_prs)],
            })

            self.simulations['qalys'].append(np.sum(qalys_discount_prs, axis=1))
            self.simulations['lyrs'].append(np.sum(lyrs_discount_prs, axis=1))
            self.simulations['costs'].append((np.sum(cost_discount_prs, axis=1)+np.mean(cost_screening_prs)))
            self.simulations['pca_deaths'].append(np.sum(deaths_pca_prs, axis=1))
            self.simulations['pca_deaths_sc'].append(np.sum(pca_death_sc, axis=1))
            self.simulations['pca_deaths_ns'].append(np.sum(pca_death_ns, axis=1))
            self.simulations['pca_deaths_nos'].append(np.sum(pca_death_nos, axis=1))
            self.simulations['pca_deaths_nos_sc'].append(np.sum(pca_death_nos_sc, axis=1))
            self.simulations['deaths_other'].append(np.sum(deaths_other_prs, axis=1))
            self.simulations['deaths_other_sc'].append(np.sum(deaths_sc_other_prs, axis=1))
            self.simulations['deaths_other_ns'].append(np.sum(deaths_ns_other_prs, axis=1))
            self.simulations['deaths_other_nos'].append(np.sum(deaths_nos_other_prs, axis=1))
            self.simulations['deaths_other_nos_sc'].append(np.sum(deaths_nos_sc_other_prs, axis=1))
            self.simulations['pca_cases'].append(np.sum(cases_prs, axis=1))
            self.simulations['pca_cases_sc'].append(np.sum(pca_incidence_sc, axis=1))
            self.simulations['pca_cases_ns'].append(np.sum(pca_incidence_ns, axis=1))
            self.simulations['pca_cases_nos'].append(np.sum(pca_incidence_nos, axis=1))
            self.simulations['pca_cases_nos_sc'].append(np.sum(pca_incidence_nos_sc, axis=1))
            self.simulations['post_screening_cases'].append(np.sum((pca_incidence_post_screening+pca_incidence_nos_sc_post_screening), axis=1))
            self.simulations['post_screening_cases_sc'].append(np.sum(pca_incidence_post_screening, axis=1))
            self.simulations['post_screening_cases_nos_sc'].append(np.sum(pca_incidence_nos_sc_post_screening, axis=1))
            self.simulations['cost_mri'].append(np.sum(cost_mri_discount_prs, axis=1))
            self.simulations['cost_biopsy'].append(np.sum(cost_biopsy_discount_prs, axis=1))
            self.simulations['n_mri'].append(np.sum(n_mri_prs, axis=1))
            self.simulations['n_biopsies'].append(np.sum(n_biopsies_prs, axis=1))
            self.simulations['n_psa_tests'].append(np.sum(n_psa_tests_prs, axis=1))
            self.simulations['overdiagnosis'].append(np.sum(overdiagnosis_prs, axis=1))
            self.simulations['overdiagnosis_sc'].append(np.sum(overdiagnosis_sc, axis=1))
            self.simulations['overdiagnosis_nos_sc'].append(np.sum(overdiagnosis_nos_sc, axis=1))

            self.cohort_list.append(cohort)
            self.outcomes_list.append(outcomes)

        return self.cohort_list, self.outcomes_list, self.simulations
