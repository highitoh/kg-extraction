---
name: confluence-extraction
description: Test Confluence page knowledge extraction end-to-end (use when user asks to test, run, or verify Confluence extraction)
---

# Confluence Extraction Test

You are an agent for testing Confluence page knowledge extraction end-to-end in one pass.

## Required Information

Ask the user for the following if not already provided:
- **Confluence page URL** (e.g., `https://yourcompany.atlassian.net/wiki/spaces/SPACE/pages/123456/Page+Title`)
- **Project name** (default: `test-confluence-YYYYMMDD` using today's date)

## Execution Steps

### 1. Run Extraction

```bash
cd /workspace/app && ./extract.sh "CONFLUENCE_URL" --project PROJECT_NAME
```

The default model, `gemini-3.5-flash-lite`, keeps testing cheap.

### 2. Verify Output Files

```bash
ls -lh output/PROJECT_NAME/
cat output/PROJECT_NAME/knowledge.json | jq '{classes: (.classes | length), properties: (.properties | length)}'
```

### 3. Check Logs for Errors

```bash
tail -80 log/extract_workflow.log
```

### 4. Report Results

```
### Confluence Extraction Result
- URL: `[Confluence page URL]`
- Project: `[project name]`
- Status: ✅ PASS / ❌ FAIL
- Classes extracted: X
- Properties extracted: X
- Issues: [None / Description]
```

## Notes

- Credentials are read from environment variables: `ATLASSIAN_EMAIL`, `ATLASSIAN_API_TOKEN`, `ATLASSIAN_INSTANCE_DOMAIN`
- Supported URL formats:
  - Full page URL: `https://instance.atlassian.net/wiki/spaces/SPACE/pages/PAGE_ID/Title`
  - Full space URL: `https://instance.atlassian.net/wiki/spaces/SPACE`
  - Shorthand page: `confluence:instance.atlassian.net:SPACE:PAGE_ID`
  - Shorthand space: `confluence:instance.atlassian.net:SPACE`
- If credentials are missing, report clearly which environment variables need to be set
- If extraction fails, check the log and report the root cause
