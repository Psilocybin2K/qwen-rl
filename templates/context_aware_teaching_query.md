# Context-Aware Teaching Query Template

Generate test steps for the following query within the specified business context to create a training example:

**Business Context**: {context}

**Query**: {query}

**Instructions**:

- Consider the business context when generating test steps for training
- Provide clear, actionable test steps relevant to the specific domain
- Respond with a JSON array of strings
- Each step should be specific and unambiguous
- Maintain logical sequence appropriate for the business context
- Focus on user actions and system responses within the given context
- Ensure steps are relevant to the specific business domain
- Provide consistent, high-quality examples for model learning

**Response Format**:

```json
["Step 1", "Step 2", "Step 3"]
```

**Training Focus**:

- Generate examples that demonstrate best practices for the specific domain
- Ensure consistency with established patterns within the business context
- Provide clear, unambiguous instructions relevant to the domain
- Support effective model learning and generalization for domain-specific tasks

**Context Guidelines**:

- Adapt terminology and processes to match the business domain
- Consider domain-specific workflows and requirements
- Ensure steps are practical within the given business context
- Maintain consistency with industry standards for the domain
