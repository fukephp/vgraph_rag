---
name: verify-concept
description: Stamp OKF verified frontmatter on a concept. Use when a human has reviewed a knowledge concept and wants to record human:<id> verification.
---

# Verify an OKF concept

Edit the concept file in place. Do not run Python.

1. Identify the concept `.md` (user path, or the open file). Skip `index.md` and `log.md`.
2. Resolve the actor:
   - If the user gave an id, use `human:<id>` (add the `human:` prefix if missing).
   - Else run `git config user.name`, slugify (lowercase, spaces to `-`), and use `human:<slug>`.
3. `at` is now in UTC ISO 8601: `YYYY-MM-DDTHH:MM:SSZ`.
4. In YAML frontmatter:
   - No `verified` → add `verified: { by: human:<id>, at: <utc> }`
   - `verified` is a mapping `{ by, at }` → turn it into a list and append the new event
   - `verified` is already a list → append `{ by: human:<id>, at: <utc> }`
5. Do not change `generated`, body, or other keys. Do not remove earlier `verified` entries.
