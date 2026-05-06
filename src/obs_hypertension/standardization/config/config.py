# standardization_code_process/config.py

MAX_THERAPIES = 2

MAX_OUTCOMES = 2

MISSING_VALUE = "not specified"

COLUMN_ORDER = [

            'corpusid',
            'title',
            'year',
            'doi',
            'study_design',
            'group_label',
            'sample_size',

            'drug_1','route_1','dose_1','frequency_1','duration_1',
            'drug_2','route_2','dose_2','frequency_2','duration_2',

            'cohort_description',
            'maternal_clinical_diagnosis',
            'cohort_stratification',
            'maternal_age_range',
            'gestational_age_range',
            'systolic_pressure_range',
            'diastolic_pressure_range',

            'maternal_outcome_name_1',
            'maternal_outcome_metric_1',
            'maternal_outcome_value_1',
            'maternal_outcome_unit_1',
            'maternal_outcome_type_1',
            'maternal_outcome_comparison_1',
            'maternal_outcome_p_value_1',

            'maternal_outcome_name_2',
            'maternal_outcome_metric_2',
            'maternal_outcome_value_2',
            'maternal_outcome_unit_2',
            'maternal_outcome_type_2',
            'maternal_outcome_comparison_2',
            'maternal_outcome_p_value_2',

            'fetal_outcome_name_1',
            'fetal_outcome_metric_1',
            'fetal_outcome_value_1',
            'fetal_outcome_unit_1',
            'fetal_outcome_type_1',
            'fetal_outcome_comparison_1',
            'fetal_outcome_p_value_1',

            'fetal_outcome_name_2',
            'fetal_outcome_metric_2',
            'fetal_outcome_value_2',
            'fetal_outcome_unit_2',
            'fetal_outcome_type_2',
            'fetal_outcome_comparison_2',
            'fetal_outcome_p_value_2',

            'anomaly_outcome_name_1',
            'anomaly_outcome_metric_1',
            'anomaly_outcome_value_1',
            'anomaly_outcome_unit_1',
            'anomaly_outcome_type_1',
            'anomaly_outcome_comparison_1',
            'anomaly_outcome_p_value_1',

            'anomaly_outcome_name_2',
            'anomaly_outcome_metric_2',
            'anomaly_outcome_value_2',
            'anomaly_outcome_unit_2',
            'anomaly_outcome_type_2',
            'anomaly_outcome_comparison_2',
            'anomaly_outcome_p_value_2',

            'main_findings',
            'winner',
            'winner_reason',
            'statistical_significance_vs_control'

            ]