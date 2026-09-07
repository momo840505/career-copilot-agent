# Security

## Deployment boundary

The public service exposes FastAPI. ChromaDB is used through an embedded
`PersistentClient`; the Chroma HTTP management API is not exposed.

The shared access code limits demo usage. It is not a user-account or authorization
system.

## Controls

- Secrets are supplied through environment variables.
- Access-code comparison uses constant-time comparison.
- Job descriptions and revision feedback have explicit size limits.
- Review threads are tied to the browser client that created them.
- Gap-analysis citations must come from evidence retrieved for the same requirement.
- Index rebuilds replace the Chroma collection to prevent stale evidence.
- The container runs as a non-root user.
- GitHub Actions runs lint, unit tests, frontend audit/build, live evals, and a Docker
  boot smoke test.
- The default branch ruleset blocks branch deletion and non-fast-forward updates.

## ChromaDB

Published Chroma server advisories should be reviewed before changing this application
to a networked Chroma deployment. The current application does not expose Chroma's
server endpoints.
