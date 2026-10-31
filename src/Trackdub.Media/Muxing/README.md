# src/Trackdub.Media/Muxing

## Purpose

Muxing/export implementation.

## What belongs here

FFmpeg mux commands and output verification.

`FfmpegSequenceRenderer` implements the `ISequenceExportRenderer` boundary. It trims every linked
video and selected audio input to the placement's original source range, supports original source,
dubbed, or no audio, applies the shared output frame and per-placement crop, then concatenates in
sequence order. The application owns source/artifact
resolution, sequence subtitle rebasing, delivery eligibility and tier rules. The renderer validates
media stream/range coverage and writes to a temporary file so a failed render preserves the last
published output.

## What should not go here

Export business rules.

## Agent guidance

Keep changes scoped to this directory's purpose. If a task requires crossing boundaries, update the relevant architecture note or ADR first.
