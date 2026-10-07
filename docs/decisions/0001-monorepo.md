# Monorepo for AVF Ventures

## Status
Accepted

## Context

AVF needs a place for reusable framework code, shared services, venture implementations, experiments, research notes, documentation, and publication artifacts.

## Decision

Use a monorepo with top-level areas for `avf`, `ventures`, `experiments`, `research`, `docs`, `paper`, `tests`, `scripts`, and `config`.

## Alternatives

Separate repositories for each venture and framework package.

## Consequences

The monorepo makes early extraction and comparison across ventures easier. It also requires discipline to keep venture-specific code isolated until abstractions are justified.
