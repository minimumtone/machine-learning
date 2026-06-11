# Error Analysis Report

# Paper ref: Table (tab:error_analysis) -- error breakdown by method


## baseline1_llm_only

- Total queries: 100
- Execution failures: 0 (0.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 17

### Lowest accuracy queries:
- q_easy_001 (easy): acc=0.00
- q_easy_019 (easy): acc=0.00
- q_medium_002 (medium): acc=0.00
- q_medium_007 (medium): acc=0.00
- q_hard_009 (hard): acc=0.00


## baseline2_full_schema

- Total queries: 100
- Execution failures: 5 (5.0%)
- Syntax errors: 5
- Hallucinated tables: 0
- Hallucinated joins: 21

### Lowest accuracy queries:
- q_medium_002 (medium): acc=0.00
- q_medium_007 (medium): acc=0.00
- q_hard_016 (hard): acc=0.00
- q_hard_017 (hard): acc=0.00
- q_hard_018 (hard): acc=0.00


## baseline3_rule_based

- Total queries: 100
- Execution failures: 5 (5.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 3

### Lowest accuracy queries:
- q_easy_003 (easy): acc=0.00
- q_easy_012 (easy): acc=0.00
- q_easy_018 (easy): acc=0.00
- q_medium_010 (medium): acc=0.00
- q_medium_020 (medium): acc=0.00


## baseline4_fk_list

- Total queries: 100
- Execution failures: 2 (2.0%)
- Syntax errors: 2
- Hallucinated tables: 0
- Hallucinated joins: 17

### Lowest accuracy queries:
- q_easy_018 (easy): acc=0.00
- q_medium_002 (medium): acc=0.00
- q_medium_007 (medium): acc=0.00
- q_hard_017 (hard): acc=0.00
- q_hard_018 (hard): acc=0.00


## proposed

- Total queries: 100
- Execution failures: 0 (0.0%)
- Syntax errors: 0
- Hallucinated tables: 0
- Hallucinated joins: 2

### Lowest accuracy queries:
- q_medium_002 (medium): acc=0.00
- q_medium_007 (medium): acc=0.00
- q_medium_025 (medium): acc=0.00
- q_hard_018 (hard): acc=0.00
- q_vhard_016 (very_hard): acc=0.00
