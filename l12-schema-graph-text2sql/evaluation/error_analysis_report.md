# Error Analysis Report


## baseline1_llm_only

- Total queries: 100
- Execution failures: 100 (100.0%)
- Syntax errors: 2
- Hallucinated tables: 100
- Hallucinated joins: 6

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00


## baseline2_full_schema

- Total queries: 100
- Execution failures: 2 (2.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 54

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_007 (easy): acc=0.00
- q_easy_010 (easy): acc=0.00
- q_easy_015 (easy): acc=0.00


## baseline3_rule_based

- Total queries: 100
- Execution failures: 5 (5.0%)
- Syntax errors: 1
- Hallucinated tables: 0
- Hallucinated joins: 95

### Lowest accuracy queries:
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_012 (easy): acc=0.00
- q_easy_017 (easy): acc=0.00
- q_easy_018 (easy): acc=0.00


## baseline4_fk_list

- Total queries: 100
- Execution failures: 93 (93.0%)
- Syntax errors: 5
- Hallucinated tables: 0
- Hallucinated joins: 94

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00


## proposed

- Total queries: 100
- Execution failures: 1 (1.0%)
- Syntax errors: 1
- Hallucinated tables: 0
- Hallucinated joins: 96

### Lowest accuracy queries:
- q_medium_002 (medium): acc=0.00
- q_hard_001 (hard): acc=0.00
- q_hard_002 (hard): acc=0.00
- q_hard_018 (hard): acc=0.00
- q_hard_023 (hard): acc=0.00
