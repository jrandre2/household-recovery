Manual parameter extraction notes (US-only studies)

- wilson_2021.yaml (Flood Recovery Outcomes Disaster Assistance Barriers Vulnerable - Wilson 2021)
  - fema_ha_max = 35500.0
    Quote: "Maximum of $35,500 for housing assistance"
  - sba_loan_max = 200000.0
    Quote: "Real property loans up to $200,000"
  Source: /Users/jesseandrews/Downloads/PDFs/Flood Recovery Outcomes Disaster Assistance Barriers Vulnerable - Wilson (2021).pdf

- affordable_housing_social_equity.yaml (Affordable Housing, Disasters, and Social Equity)
  - fema_ha_max = 33000.0
    Quote: "cap on individual assistance, which was $33,000 as of 2018."
  Source: /Users/jesseandrews/Downloads/PDFs/Affordable Housing  Disasters  and Social Equity.pdf

- wang_2021_quant_model.yaml (Quantitative modeling of residential building disaster recovery - Wang 2021, Joplin tornado)
  - insurance_penetration_rate = 0.85
    Quote: "More than 85% of homeowners have homeowner insurance in the United States"
  - transition_r2 = 0.9861
    Quote: "percentage of buildings in the study area achieving full recovery after one and a half years, three years, and four years was ... 98.61%"
  - base_recovery_rate = 0.0286
    Note: Derived from 51.49% at 1.5 years -> 0.5149 / 18 months.
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0075.txt

- hamideh_2018_housing_recovery.yaml (Housing Recovery after Disasters - Hamideh et al. 2018, Galveston Ike)
  - insurance_penetration_rate = 0.50
    Quote: "50% of homeowners did not have flood insurance"
  - sba_income_floor = 38700.0
    Quote: "median income for the urban core was $38,700"
  - cdbg_dr_coverage_rate = 0.20
    Quote: "mitigation to prevent future damage of up to 20% of the property's value"
    Note: Used as a proxy for CDBG-DR coverage rate.
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0026.txt

- frimpong_2025_principal_agent.yaml (Hazard Risk Management Principal-Agent Problem - Frimpong 2025)
  - insurance_penetration_rate = 0.57
    Quote: "57% of our sample has flood insurance on their property"
  Source: /Users/jesseandrews/Downloads/PDFs/Hazard Risk Management Principal-Agent Problem - Frimpong (2025).pdf

- levee_risk_perception_2012.yaml (Flood risk perception in lands protected by 100 year levees)
  - insurance_penetration_rate = 0.20
    Quote: "20% of respondents had purchased a flood insurance policy"
  Source: /Users/jesseandrews/Downloads/PDFs/Flood risk perception in lands protected by 100 year levees.pdf

- lee_2022_evacuation_return.yaml (Evacuation Return/Home-Switch Stability - Lee 2022, Harris County TX)
  - base_recovery_rate = 0.3333
    Quote: "more than half of the census tracts in Harris County stopped moving out of their homes within 6 weeks"
    Note: Derived as 0.5 / 1.5 months.
  - transition_r2 = 0.80
  - transition_relocate = 0.20
    Quote: "return pace of the first 80% of the population is faster than the remaining 20% ... 20% of the areas return with considerable lag"
    Note: Used as a proxy for completion vs. lag/relocation share.
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0003.txt

- cdbg_dr_2019.yaml (Housing Recovery and CDBG-DR - HUD 2019)
  - base_recovery_rate = 0.022
    Quote: "All housing activities on average took 3.8 years to complete from the time of declaration"
    Note: Converted to monthly base rate assuming 1 step = 1 month: 1 / (3.8 * 12) = 0.0219.
  Source: /Users/jesseandrews/Downloads/PDFs/HousingRecovery_CDBG-DR.pdf

- horney_2016_community_recovery.yaml (Developing Indicators Post-Disaster Community Recovery - Horney 2016)
  - No explicit RecovUS numeric parameters found; defaults used.
  Source: /Users/jesseandrews/Downloads/PDFs/Developing Indicators Post-Disaster Community Recovery US - Horney (2016).pdf

- jordan_2013_indicators.yaml (Indicators of Community Recovery - Jordan et al. 2013)
  - No explicit RecovUS numeric parameters found; defaults used.
  Source: /Users/jesseandrews/Downloads/PDFs/jordan-javernick-will-2013-indicators-of-community-recovery-content-analysis-and-delphi-approach.pdf

- punta_gorda_charley_2004.yaml (Punta Gorda, FL - Hurricane Charley)
  - base_recovery_rate = 0.0375
    Quote: "housing recovery was roughly 90 per cent complete after two years and 100 per cent complete after three years"
    Note: Derived as 0.90 / 24 months.
  - transition_r2 = 0.92
    Quote: "The vast majority (92 per cent) of buildings were damaged then repaired."
  - transition_relocate = 0.08
    Note: Derived as 1 - 0.92 (proxy for non-repaired share).
  - sba_income_floor = 32460.0
    Quote: "average annual per capita income of USD 32,460"
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0002.txt

- camp_fire_tubbs_2017_2018.yaml (Tubbs/Camp Fires, CA)
  - insurance_penetration_rate = 0.84
    Quote: "Insurance penetration [%] ... 84"
  - sba_income_floor = 51396.0
    Quote: "Median household income [$] ... 51,396"
  - transition_r2 = 0.09
    Quote: "Parcels rebuilt [% of destroyed] ... 9"
  - base_recovery_rate = 0.0019
    Note: Derived as 0.09 / 47 months.
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0006.txt

- west_tx_explosion_2013.yaml (West, TX fertilizer plant explosion)
  - fema_ha_max = 35000.0
    Quote: "FEMA assistance only covers what insurance does not and is capped near $35,000 per household"
  - transition_r2 = 0.82
    Quote: "82 percent of houses in our analysis completely repaired or reconstructed."
  - transition_relocate = 0.11
    Quote: "Almost 11 percent of audited parcels remained cleared lots three years after the explosion."
  - base_recovery_rate = 0.0228
    Note: Derived as 0.82 / 36 months.
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0013.txt

- housing_type_matters_ike_2021.yaml (Housing type matters for pace of recovery - Hurricane Ike)
  - base_recovery_rate = 0.0333
    Quote: "Housing recovery is usually characterized at an aggregate level as being completed within two to three years after the event"
    Note: Derived as 1 / 30 months.
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0027.txt

- disaster_disparities_new_orleans_2010.yaml (Disaster disparities - New Orleans, Katrina)
  - transition_r2 = 0.719
    Quote: "return rate of 71.9%"
  - transition_relocate = 0.1002
    Quote: "37,839 properties were supported ... while the state buyout ... numbered 4,212"
    Note: Derived as 4212 / (4212 + 37839).
  - sba_income_floor = 48145.0
    Quote: "average household income of $48,145"
  - base_recovery_rate = 0.0200
    Note: Derived as 0.719 / 36 months.
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0031.txt

- hurricane_andrew_1992.yaml (Hurricane Andrew - Miami-Dade, FL)
  - fema_ha_max = 5000.0
    Quote: "FEMA's Minimal Home Repair Program (MHR) provides small grants, usually up to $5,000"
  - insurance_penetration_rate = 0.93
    Quote: "Housing reconstruction ... funded primarily (93%) by insurance settlements"
  - sba_uptake_rate = 0.07
    Quote: "supplemental funding (less than 7%) from SBA's loan program"
    Note: Used as a proxy for SBA uptake.
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0036.txt

- katrina_stable_housing_2022.yaml (Time to stable housing after Katrina)
  - transition_r0 = 0.1867
    Quote: "Stable housing within 18 months ... 18.67"
  - transition_r2 = 0.6618
    Quote: "Overall ... 66.18"
  - sba_income_floor = 20000.0
    Quote: "Below $20,000"
  - base_recovery_rate = 0.0139
    Quote: "median time to stable housing was 1082 days"
    Note: Derived as 0.5 / 36 months.
  Source: /Volumes/T9/Projects/Housing Benchmark Papers/data_raw/txt/ST0056.txt
