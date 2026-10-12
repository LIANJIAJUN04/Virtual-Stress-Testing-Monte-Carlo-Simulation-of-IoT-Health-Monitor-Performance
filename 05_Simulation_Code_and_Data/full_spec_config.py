FULL_SPEC_SETTINGS = {
    'TOTAL_RUNS': 1000,
    'RUNS_PER_BATCH': 100,
    'SIMULATION_HOURS': 6,
    'TIME_STEP_SECONDS': 2.0,
    'ENVIRONMENTS': ['Controlled', 'Residential'],
    'OUTPUT_DIR': 'full_spec_results'
}

TEST_SETTINGS = {
    'TOTAL_RUNS': 10,
    'RUNS_PER_BATCH': 5,
    'SIMULATION_HOURS': 2,
    'TIME_STEP_SECONDS': 5.0,
    'ENVIRONMENTS': ['Controlled'],
    'OUTPUT_DIR': 'test_results'
}

USE_TEST_MODE = False

if USE_TEST_MODE:
    CONFIG = TEST_SETTINGS
else:
    CONFIG = FULL_SPEC_SETTINGS
