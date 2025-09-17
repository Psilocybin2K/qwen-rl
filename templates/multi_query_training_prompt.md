# Multi-Query Training Coordination Template

You are coordinating a multi-query training session for a QA test case generation system.

## Training Session Overview

**Session ID**: {session_id}
**Total Queries**: {total_queries}
**Current Query**: {current_query_index}/{total_queries}
**Business Domain**: {business_domain}

## Current Query Details

**Query**: {query}
**Context**: {context}
**Expected Answer**: {correct_answer}

## Training Instructions

1. **Generate Test Steps**: Create clear, actionable test steps for the given query
2. **Context Awareness**: Consider the business context when generating steps
3. **Consistency**: Maintain consistency with previous training examples
4. **Quality**: Ensure high-quality examples for effective model learning

## Output Format

Respond with a JSON array of test steps:

```json
["Step 1", "Step 2", "Step 3"]
```

## Training Guidelines

- **Domain Relevance**: Ensure steps are relevant to the business domain
- **Actionability**: Each step should be specific and actionable
- **Logical Flow**: Maintain logical sequence appropriate for the context
- **Consistency**: Follow established patterns from previous examples
- **Quality**: Provide examples that demonstrate best practices

## Progress Tracking

- **Completed Queries**: {completed_queries}
- **Successful Queries**: {successful_queries}
- **Current Success Rate**: {success_rate}%

Continue with the current query following the established training patterns.
