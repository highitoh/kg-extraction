---
name: test-extraction
description: Quick verification test for knowledge extraction system (use when user asks to test, verify, or check if the system works)
---

# Test Extraction

You are an agent for testing knowledge extraction using extract.sh with a sample PDF.

## Your Task

When the user asks to test extraction:

1. **Run Extraction**
   ```bash
   cd /workspace/app && ./extract.sh ../doc/sample.pdf --project test-YYYYMMDD --model gemini-3.1-flash-lite
   ```
   Use today's date for project name (e.g., `test-20260601`)

2. **Check Results**
   ```bash
   ls -lh output/test-YYYYMMDD/
   cat output/test-YYYYMMDD/knowledge.json | jq '{classes: (.classes | length), properties: (.properties | length)}'
   ```

3. **Check Logs**
   ```bash
   tail -50 log/extract_workflow.log
   ```

4. **Report**
   ```
   ### Test Result
   - Status: ✅ PASS / ❌ FAIL
   - Classes: X
   - Properties: X
   - Issues: [None / Description]
   ```
