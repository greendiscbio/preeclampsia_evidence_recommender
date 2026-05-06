import re

PRIMARY_DRUGS = ["labetalol", "methyldopa", "nifedipine", "hydralazine", "sulfate", "magnesium"]

PREGNANCY_REGEX = r"""(?i)\b(
    pregnan\w+|gestation\w*|antenatal|perinatal|
    maternal|fetal|foetal|obstetric|ob-gyn|obstetrics|
    gravida|gravida\w*|gravidity|nulliparous|
    multiparous|parous|preeclampsia|pre[- ]?eclampsia|
    eclampsia|hellp|iugr|sga|birth\s*outcome
)\b"""

HYPERTENSION_REGEX = r"""(?i)\b(
    hypertension|hypertensive|high\s*blood\s*pressure|
    blood[- ]?pressure|bp\b|htn\b|
    gestational\shypertension|pregnancy[- ]?induced\shypertension|
    pih|antihypertens\w+|beta[- ]?blocker|
    ace\sinhibitor|angiotensin\sreceptor\sblocker|
    arb|calcium\schannel\sblocker|
    diuretic|vasodilator
)\b"""
