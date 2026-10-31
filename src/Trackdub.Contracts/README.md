# src/Trackdub.Contracts

## Purpose

Stable DTOs/contracts.

## What belongs here

Cross-boundary records if needed.

The `Editing` contracts include `ISequenceExportRenderer` and immutable render plans. The application
resolves one media input per stable sequence placement and explicitly selects original source audio,
dubbed audio, or no audio. Dubbed audio paths are full-source timelines aligned to source time zero
and must be available for every placement in the export.

## What should not go here

Domain logic or implementation.

## Agent guidance

Keep changes scoped to this directory's purpose. If a task requires crossing boundaries, update the relevant architecture note or ADR first.
