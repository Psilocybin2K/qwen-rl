# User Query Template

Generate test steps for the following query:

**Query**: {query}

**Context**: {context}

**Instructions**:

- Provide clear, actionable test steps based on the context
- Respond with a JSON array of strings
- Each step should be specific and unambiguous
- Maintain logical sequence
- Focus on user actions and system responses

**Response Format**:

```json
["Step 1", "Step 2", "Step 3"]
```

