# Use cases — results for every project

`data/use_cases/` holds 66 generated realistic projects (12 industries, 30–60 named tasks, tight resources; see `scripts/make_use_cases.py`). They were generated once with fixed parameters and are not part of training. All six solvers ran on every project with the same budget (budget and seed set automatically per project). Produced by `scripts/evaluate_use_cases.py`; raw data in `docs/use_cases.csv`.

- **TS or PSS won 64/66** projects (97.0%): TS 12, PSS 52.
- 40 of these wins are a strictly shorter makespan than every baseline; on average the best proposed makespan is 1.77 periods shorter than the best baseline's.
- All schedules feasible on 66/66 projects.
- Projects not won by TS/PSS: shipbuilding_16.json (GA, total finish time), software_02.sm (SA, makespan).

| Project | Tasks | Lower bound | TS | PSS | GA | SA | BCO | PRH | Winner | Decided by |
|---|---|---|---|---|---|---|---|---|---|---|
| aerospace_07.json | 56 | 56 | **64** | 65 | 70 | 76 | 74 | 69 | TS | makespan |
| aerospace_19.json | 59 | 51 | **56** | 57 | 59 | 60 | 64 | 59 | TS | makespan |
| aerospace_31.json | 58 | 46 | 50 | **49** | 50 | 50 | 52 | 52 | PSS | makespan |
| aerospace_43.json | 54 | 51 | **60** | **60** | 65 | 63 | 67 | 64 | PSS | total finish time |
| aerospace_55.json | 31 | 40 | **50** | **50** | **50** | **50** | 52 | **50** | PSS | reliability (mean evaluations over 5 seeds) |
| construction_01.json | 33 | 58 | **89** | **89** | **89** | **89** | 90 | **89** | PSS | reliability (mean evaluations over 5 seeds) |
| construction_13.json | 32 | 49 | **66** | **66** | **66** | 67 | 72 | **66** | TS | total finish time |
| construction_25.json | 48 | 56 | 75 | **72** | 77 | 78 | 81 | 80 | PSS | makespan |
| construction_37.json | 58 | 57 | 79 | **77** | 78 | 79 | 85 | 78 | PSS | makespan |
| construction_49.json | 50 | 56 | 74 | **73** | 80 | 75 | 81 | 75 | PSS | makespan |
| construction_61.json | 50 | 56 | 72 | **70** | 72 | 72 | 77 | 72 | PSS | makespan |
| data_centre_05.sm | 39 | 45 | 55 | **53** | 56 | 54 | 60 | 56 | PSS | makespan |
| data_centre_17.sm | 54 | 52 | 61 | **60** | 64 | 64 | 66 | 63 | PSS | makespan |
| data_centre_29.sm | 36 | 43 | 61 | **57** | 59 | 61 | 63 | 58 | PSS | makespan |
| data_centre_41.sm | 52 | 48 | 53 | **52** | 55 | 55 | 58 | 55 | PSS | makespan |
| data_centre_53.sm | 31 | 39 | **50** | **50** | **50** | **50** | 52 | 51 | TS | total finish time |
| data_centre_65.sm | 37 | 37 | **48** | **48** | 49 | 49 | 50 | 50 | TS | total finish time |
| event_11.sm | 58 | 44 | 53 | **52** | 56 | 56 | 59 | 56 | PSS | makespan |
| event_23.sm | 39 | 41 | **53** | **53** | 56 | 57 | 58 | 56 | TS | total finish time |
| event_35.sm | 48 | 41 | 56 | **53** | 56 | 58 | 62 | 57 | PSS | makespan |
| event_47.sm | 41 | 44 | **62** | **62** | **62** | 64 | 70 | 63 | PSS | total finish time |
| event_59.sm | 43 | 48 | **59** | **59** | 62 | 63 | 67 | 61 | PSS | total finish time |
| film_09.rcp | 32 | 43 | **59** | **59** | 60 | 60 | 60 | 60 | TS | total finish time |
| film_21.rcp | 47 | 45 | 53 | **52** | 58 | 58 | 58 | 55 | PSS | makespan |
| film_33.rcp | 34 | 46 | 59 | **58** | 59 | 59 | 64 | 59 | PSS | makespan |
| film_45.rcp | 42 | 38 | 46 | **45** | 47 | 46 | 48 | 47 | PSS | makespan |
| film_57.rcp | 53 | 44 | 58 | **56** | 60 | 58 | 63 | 60 | PSS | makespan |
| hospital_06.rcp | 51 | 51 | 60 | **58** | 62 | 63 | 68 | 64 | PSS | makespan |
| hospital_18.rcp | 45 | 43 | 52 | **50** | 56 | 56 | 58 | 54 | PSS | makespan |
| hospital_30.rcp | 35 | 34 | 45 | **44** | 47 | 46 | 49 | 47 | PSS | makespan |
| hospital_42.rcp | 32 | 48 | 64 | **61** | 62 | 66 | 64 | 63 | PSS | makespan |
| hospital_54.rcp | 44 | 36 | 46 | **45** | 48 | 47 | 50 | 49 | PSS | makespan |
| hospital_66.rcp | 39 | 37 | **43** | **43** | **43** | 46 | 47 | **43** | PSS | reliability (mean evaluations over 5 seeds) |
| manufacturing_03.rcp | 41 | 41 | 54 | **51** | 56 | 55 | 58 | 54 | PSS | makespan |
| manufacturing_15.rcp | 42 | 46 | 60 | **57** | 61 | 61 | 65 | 60 | PSS | makespan |
| manufacturing_27.rcp | 60 | 57 | 64 | **62** | 68 | 68 | 71 | 66 | PSS | makespan |
| manufacturing_39.rcp | 52 | 44 | 53 | **51** | 52 | 54 | 55 | 53 | PSS | makespan |
| manufacturing_51.rcp | 36 | 38 | **49** | **49** | 50 | 50 | 55 | 50 | TS | reliability (mean evaluations over 5 seeds) |
| manufacturing_63.rcp | 48 | 44 | 57 | **55** | 57 | 57 | 60 | 58 | PSS | makespan |
| oil_gas_10.json | 56 | 52 | 60 | **58** | 64 | 64 | 65 | 62 | PSS | makespan |
| oil_gas_22.json | 50 | 42 | 56 | **54** | 56 | 57 | 58 | **54** | PSS | total finish time |
| oil_gas_34.json | 43 | 49 | **57** | **57** | 61 | 59 | 64 | 61 | PSS | total finish time |
| oil_gas_46.json | 30 | 33 | 44 | **43** | **43** | 44 | 45 | **43** | PSS | total finish time |
| oil_gas_58.json | 30 | 42 | **50** | **50** | **50** | 52 | 53 | **50** | PSS | reliability (mean evaluations over 5 seeds) |
| pharma_12.rcp | 38 | 45 | 64 | **60** | 61 | 61 | 63 | 61 | PSS | makespan |
| pharma_24.rcp | 46 | 55 | **63** | **63** | 68 | 67 | 69 | 68 | PSS | reliability (mean evaluations over 5 seeds) |
| pharma_36.rcp | 41 | 43 | **61** | **61** | 64 | 65 | **61** | 62 | TS | total finish time |
| pharma_48.rcp | 58 | 41 | 49 | **47** | 50 | 53 | 54 | 52 | PSS | makespan |
| pharma_60.rcp | 32 | 40 | **49** | **49** | **49** | **49** | **49** | **49** | PSS | reliability (mean evaluations over 5 seeds) |
| rail_08.sm | 43 | 42 | **49** | **49** | 50 | 54 | 54 | 50 | TS | total finish time |
| rail_20.sm | 58 | 48 | 56 | **55** | 59 | 60 | 63 | 58 | PSS | makespan |
| rail_32.sm | 60 | 56 | 62 | **61** | 67 | 67 | 73 | 64 | PSS | makespan |
| rail_44.sm | 41 | 44 | 52 | **50** | 51 | 53 | 54 | 51 | PSS | makespan |
| rail_56.sm | 45 | 43 | 51 | **50** | 54 | 51 | 57 | 54 | PSS | makespan |
| shipbuilding_04.json | 53 | 46 | 59 | **58** | 61 | 60 | 67 | 61 | PSS | makespan |
| shipbuilding_16.json | 31 | 42 | **48** | **48** | **48** | **48** | 50 | **48** | GA | total finish time |
| shipbuilding_28.json | 39 | 54 | **64** | **64** | 66 | **64** | 68 | 68 | TS | total finish time |
| shipbuilding_40.json | 50 | 41 | 47 | **46** | 50 | 50 | 55 | 50 | PSS | makespan |
| shipbuilding_52.json | 41 | 40 | 52 | **49** | **49** | 50 | 54 | 51 | PSS | total finish time |
| shipbuilding_64.json | 59 | 46 | 49 | **48** | 52 | 52 | 55 | 50 | PSS | makespan |
| software_02.sm | 33 | 41 | 63 | 63 | 63 | **62** | 63 | 63 | SA | makespan |
| software_14.sm | 60 | 45 | **51** | 52 | 56 | 55 | 60 | 54 | TS | makespan |
| software_26.sm | 36 | 50 | 59 | **57** | 60 | 59 | 64 | **57** | PSS | total finish time |
| software_38.sm | 56 | 51 | 67 | **65** | 68 | 70 | 71 | 69 | PSS | makespan |
| software_50.sm | 59 | 54 | 63 | **62** | 65 | 65 | 68 | 66 | PSS | makespan |
| software_62.sm | 34 | 50 | **65** | **65** | 69 | 69 | 70 | **65** | PSS | reliability (mean evaluations over 5 seeds) |
