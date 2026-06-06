# Error Analysis Report


## baseline1_llm_only

- Total queries: 100
- Execution failures: 100 (100.0%)
- Syntax errors: 0
- Hallucinated tables: 12
- Hallucinated joins: 0

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00


## baseline2_full_schema

- Total queries: 100
- Execution failures: 74 (74.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 2

### Lowest accuracy queries:
- q_easy_002 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_010 (easy): acc=0.00
- q_easy_019 (easy): acc=0.00
- q_easy_020 (easy): acc=0.00


## baseline3_rule_based

- Total queries: 100
- Execution failures: 3 (3.0%)
- Syntax errors: 1
- Hallucinated tables: 0
- Hallucinated joins: 94

### Lowest accuracy queries:
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_012 (easy): acc=0.00
- q_easy_017 (easy): acc=0.00
- q_easy_018 (easy): acc=0.00


## baseline4_fk_list

- Total queries: 100
- Execution failures: 98 (98.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 0

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00


## proposed

- Total queries: 100
- Execution failures: 3 (3.0%)
- Syntax errors: 3
- Hallucinated tables: 0
- Hallucinated joins: 92

### Lowest accuracy queries:
- q_easy_009 (easy): acc=0.00
- q_easy_017 (easy): acc=0.00
- q_easy_018 (easy): acc=0.00
- q_medium_007 (medium): acc=0.00
- q_medium_029 (medium): acc=0.00
