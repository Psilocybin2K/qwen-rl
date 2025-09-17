# Context-Aware Few-Shot Example Template

Generate test steps for the following query within the specified business context, using the provided example as a reference:

**Business Context**: {context}

**Example**:
Q: {example_query}
A: {example_answer}

**Current Query**: {query}

**Instructions**:

- Use the provided example to understand the expected format and style
- Consider the business context when generating test steps
- Follow the exact format and style shown in the example
- Provide clear, actionable test steps relevant to the specific domain
- Respond with a JSON array of strings
- Each step should be specific and unambiguous
- Maintain logical sequence appropriate for the business context
- Focus on user actions and system responses within the given context

**Response Format**:

```json
["Step 1", "Step 2", "Step 3"]
```

**Context Guidelines**:

- Adapt terminology and processes to match the business domain
- Consider domain-specific workflows and requirements
- Ensure steps are practical within the given business context
- Maintain consistency with industry standards for the domain
- Follow the style and format demonstrated in the example
