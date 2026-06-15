---
skill: evaluate-extraction
description: Evaluate knowledge extraction results and propose extraction rule improvements
args: RESULT_PATH
---

# Knowledge Extraction Evaluation

Guide Claude through analyzing an extraction result and proposing improvements to extraction rules.

**IMPORTANT**: This skill proposes changes to extraction rules (`extract_rules`) and prohibition rules (`prohibit_rules`) only. It does NOT modify the metamodel structure itself (classes, properties, relationships definitions).

## Usage

```bash
/evaluate-extraction output/myproject/extraction_result_20260611_120000.json
```

## Process

**1. Load and Understand Context**
- Read the specified extraction result JSON file
- Read `/workspace/app/metamodel/metamodel.json` to understand:
  - Current metamodel structure (classes, properties, relationships)
  - Current extraction rules (`extract_rules`)
  - Current prohibition rules (`prohibit_rules`)
- If the result references a project, read that project's `knowledge.json` too
- If project is not specified in the result path or unclear, ask the user which project should be analyzed

**2. Analyze Extraction Quality**

Examine the extracted knowledge for:
- **Entity coverage**: Which classes were used? How many of each?
- **Relationship patterns**: Are entities properly connected? Missing links?
- **Property completeness**: Are required properties present? Are descriptions meaningful?
- **Classification accuracy**: Do entities fit their assigned classes?
- **Depth and detail**: Is extraction shallow or comprehensive?

**3. Identify Problems**

Look for:
- Entities with missing critical properties
- Orphaned entities (no relationships)
- Misclassified entities
- Relationship types that should exist but don't
- Patterns the metamodel doesn't capture
- Classes that are too broad or too narrow

**4. Generate Evaluation Report**

Present findings in this structure:

```
📊 EXTRACTION EVALUATION
========================

SOURCE
------
File: [path]
Project: [name]
Document: [source document name]

SUMMARY STATISTICS
------------------
- Total entities: X
- Total relationships: Y
- Classes used: Z
- Validation errors: N

ENTITY DISTRIBUTION
-------------------
[Table showing count and percentage per class]

QUALITY ASSESSMENT
------------------
✓ Strengths:
  - [What was done well]

✗ Issues Found:
  1. [Issue category] (Priority: HIGH/MEDIUM/LOW)
     - [Specific examples]
     - Impact: [why this matters]

IMPROVEMENT RECOMMENDATIONS
---------------------------
[Numbered list of actionable improvements]

For each recommendation:
- What to change (extract_rules or prohibit_rules)
- Why it will help
- Example JSON modification
```

**5. Propose Extraction Rule Changes**

For each significant issue, suggest changes to `extract_rules` or `prohibit_rules` in metamodel.json.

**Rule Design Principles**:
- Create **generic, reusable rules** applicable across different documents and contexts
- Avoid overly specific rules targeting particular classes or properties
- Focus on **patterns and behaviors** rather than individual cases
- Rules should be at an appropriate abstraction level - not too narrow, not too broad

Show the actual JSON changes that would be made.

**6. Offer Next Steps**

Ask the user:
- Would you like me to apply any of these extraction rule changes?
- Should I show detailed examples of specific issues?
- Would you like to evaluate another result for comparison?

## Notes

- Be specific with examples from the actual extraction
- Prioritize issues by impact
- Keep tone constructive and actionable
- Propose **generic, reusable rules** at appropriate abstraction level
- Do NOT create overly specific rules for individual classes or properties
- Do NOT propose changes to metamodel structure (classes, properties, relationships)
- Do NOT propose prompt changes (unless directly related to extraction rules)
- Report validation errors if present
