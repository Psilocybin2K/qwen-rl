# Dataset Validation and Management Template

You are validating and managing a dataset for QA test case generation training.

## Dataset Information

**Dataset Name**: {dataset_name}
**Total Entries**: {total_entries}
**Validation Status**: {validation_status}

## Entry Validation Criteria

### Required Fields
- **query**: Must be a non-empty string describing a test scenario
- **context**: Must be a non-empty string providing business context
- **answer**: Must be a valid JSON array of test steps

### Validation Rules
1. **Query Format**: 
   - Must be a clear, specific question or request
   - Should describe a testable business process
   - Must be actionable and unambiguous

2. **Context Quality**:
   - Must provide relevant business domain information
   - Should include necessary background for understanding the query
   - Must be specific enough to guide test step generation

3. **Answer Format**:
   - Must be a valid JSON array
   - Each element must be a non-empty string
   - Steps must be actionable and logically ordered
   - Must be relevant to the query and context

## Validation Results

**Valid Entries**: {valid_entries}
**Invalid Entries**: {invalid_entries}
**Validation Errors**: {validation_errors}

## Quality Metrics

- **Completeness**: {completeness_score}%
- **Consistency**: {consistency_score}%
- **Relevance**: {relevance_score}%

## Recommendations

{recommendations}

## Next Steps

{next_steps}
