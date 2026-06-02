# Error Analysis Report


## baseline1_llm_only

- Total queries: 100
- Execution failures: 100 (100.0%)
- Syntax errors: 0
- Hallucinated tables: 7
- Hallucinated joins: 0

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00


## baseline2_full_schema

- Total queries: 100
- Execution failures: 78 (78.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 0

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_005 (easy): acc=0.00


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
- Execution failures: 97 (97.0%)
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
- Execution failures: 5 (5.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 95

### Lowest accuracy queries:
- q_easy_002 (easy): acc=0.00
- q_easy_003 (easy): acc=0.00
- q_easy_004 (easy): acc=0.00
- q_easy_006 (easy): acc=0.00
- q_easy_007 (easy): acc=0.00
