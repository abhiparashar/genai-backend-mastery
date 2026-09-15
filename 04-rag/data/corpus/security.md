# Security Policy

## Access Control

Production database access requires a just-in-time access request, which
grants credentials for a maximum of 4 hours. All production access is logged
and reviewed monthly.

## Secrets

API keys and credentials must never be committed to source control. Secrets
are stored in HashiCorp Vault and injected at deploy time.

Any key committed to a repository must be treated as compromised and rotated
immediately, even if the commit was reverted.

## Incident Response

Security incidents must be reported to the security channel within 1 hour of
discovery. The on-call security engineer acknowledges within 15 minutes.
