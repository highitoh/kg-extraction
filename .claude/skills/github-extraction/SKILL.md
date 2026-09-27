---
name: github-extraction
description: GitHub repository knowledge extraction
---

# GitHub Extraction

You are an agent for GitHub knowledge extraction functionality.

## Required Information

- GitHub Repository URL
- Project Name
- Timing (optional): Tag or Date for historical extraction

## Execution Steps

1. Run extraction
   ```bash
   # Latest state
   cd /workspace/app && ./extract.sh "GITHUB_URL" --project PROJECT-NAME
   
   # Specific date
   cd /workspace/app && ./extract.sh "GITHUB_URL" --project PROJECT-NAME --as-of YYYYMMDD
   ```

2. Verify outputs
   ```bash
   ls -lh output/test-github-NAME/
   cat output/test-github-NAME/knowledge.json | jq '.classes | length'
   ```

3. Check logs for errors
   ```bash
   tail -100 log/extract_workflow.log
   ```

## Reporting Format

After the extraction, report in this format:

```
### Extraction
- URL: `[GitHub URL]`
- Status: ✅ PASS / ❌ FAIL
- Duration: ~X minutes
- Classes extracted: X
- Properties extracted: X
- Issues: [None / Description of any issues]
```

## Your Task

When the user asks you to perform GitHub extraction:

1. Clarify Required Information. If you are not sure, ask the user about them.
2. Execute the extraction
3. Report results in the format above
4. Provide recommendations if there are any issues
