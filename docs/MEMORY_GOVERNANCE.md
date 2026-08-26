# Memory Governance

Memory retrieval is observable through candidate/compatible/selected counts,
query signatures, rejection reasons and injected record IDs/scores. Successful
turns update per-record retrieval quality counters; unsuccessful candidates are
never silently promoted.

Long-term case memories follow this lifecycle:

```text
candidate / pending_review -> promoted / active -> deprecated or revoked
```

Candidate cases contain a query signature, parameterized SQL and validation
metadata, but are outside the retrieval pool until independent evidence and a
human approver promote them. Short-term context remains bounded and records
explicit memory conflicts when a turn contains both a historical reference and
a complete independent query.

