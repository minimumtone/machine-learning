# Error Analysis Report


## baseline1_llm_only

- Total queries: 100
- Execution failures: 100 (100.0%)
- Syntax errors: 1
- Hallucinated tables: 100
- Hallucinated joins: 4

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00


## baseline2_full_schema

- Total queries: 100
- Execution failures: 0 (0.0%)
- Syntax errors: 0
- Hallucinated tables: 2
- Hallucinated joins: 25

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00
- q_easy_006 (easy): acc=0.00


## baseline3_rule_based

- Total queries: 100
- Execution failures: 2 (2.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 94

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_006 (easy): acc=0.00


## baseline4_fk_list

- Total queries: 100
- Execution failures: 89 (89.0%)
- Syntax errors: 2
- Hallucinated tables: 2
- Hallucinated joins: 81

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00


## proposed

- Total queries: 100
- Execution failures: 18 (18.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 97

### Lowest accuracy queries:
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_006 (easy): acc=0.00
- q_easy_007 (easy): acc=0.00
