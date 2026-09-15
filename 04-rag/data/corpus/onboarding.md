# Engineering Onboarding

## First Week

New engineers are issued a laptop on day one and should complete security
training within the first 5 working days. Access to production systems is
granted only after security training is complete.

## Development Environment

We target Python 3.11 and Java 17. Local development uses Docker Compose.
The main service repository is cloned from the internal GitLab instance.

## Code Review

Every change requires at least one approving review. Changes touching billing
or authentication require two approvals, one of which must come from the
owning team.
