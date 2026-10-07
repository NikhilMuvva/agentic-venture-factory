# Deterministic Verification

## Status
Accepted

## Context

TestForge needs checks that do not depend entirely on model self-evaluation.

## Decision

Use deterministic verification alongside agent generation where rules and constraints can be checked directly.

## Alternatives

Rely only on model-based review or human review.

## Consequences

Deterministic checks improve auditability for supported constraints. They do not replace human review or cover every quality dimension.
