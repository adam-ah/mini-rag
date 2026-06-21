# Project instructions

- In code or prompts do not make any assumptions about what inputs you are working with.
- Write the failing test first, then implement the fix.
- Run all tests through `./test.sh`; never invoke test modules directly.
- Do not add code comments.
- Live AI tests must use `AI_TEST_*`, never the application's configured endpoint.
- Refinement decisions and follow-up concepts must come from the reflection model. Do not encode domain-specific aliases, relationships, or gap-detection heuristics in application code; application code may only validate that proposed queries are grounded.
