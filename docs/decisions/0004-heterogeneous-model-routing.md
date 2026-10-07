# Heterogeneous Model Routing

## Status
Proposed

## Context

AVF research questions include whether different models can be routed by task to reduce cost while maintaining quality.

## Decision

Keep model routing as a central architectural concern and preserve TestForge's existing routing work during the initial migration.

## Alternatives

Use one model for all tasks, or make model selection ad hoc inside each workflow.

## Consequences

Central routing can enable cost and reliability studies. The current project does not yet claim measured cost reductions from this approach.
