# Error Analysis Report


## baseline1_llm_only

- Total queries: 100
- Execution failures: 100 (100.0%)
- Syntax errors: 0
- Hallucinated tables: 8
- Hallucinated joins: 0

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00


## baseline2_full_schema

- Total queries: 100
- Execution failures: 73 (73.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 1

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_010 (easy): acc=0.00
- q_easy_015 (easy): acc=0.00


## baseline3_rule_based

- Total queries: 100
- Execution failures: 2 (2.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 94

### Lowest accuracy queries:
- q_easy_012 (easy): acc=0.00
- q_easy_017 (easy): acc=0.00
- q_easy_018 (easy): acc=0.00
- q_medium_010 (medium): acc=0.00
- q_medium_020 (medium): acc=0.00


## baseline4_fk_list

- Total queries: 100
- Execution failures: 95 (95.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 0

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_006 (easy): acc=0.00


## proposed

- Total queries: 100
- Execution failures: 4 (4.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 95

### Lowest accuracy queries:
- q_easy_009 (easy): acc=0.00
- q_easy_017 (easy): acc=0.00
- q_easy_018 (easy): acc=0.00
- q_medium_004 (medium): acc=0.00
- q_medium_007 (medium): acc=0.00
