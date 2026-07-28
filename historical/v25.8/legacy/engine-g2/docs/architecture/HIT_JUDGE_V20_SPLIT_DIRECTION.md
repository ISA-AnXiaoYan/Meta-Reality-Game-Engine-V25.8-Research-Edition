# Hit Judge V20 Split Direction

Date: 2026-07-03
Branch: `a-test-20260703-v20-hit-candidate-attribution-split`
Base: `e3423de V19E global person ID count gate node`

## Core Principle

V20 splits the work into two independent targets.

1. Hit candidate optimization
   - Goal: do not miss real human hits as much as possible.
   - Secondary goal: reduce false positives caused by human-motion event noise.
   - Authority: this path decides whether a hit candidate/final hit event exists.

2. Global bullet reconstruction and damage attribution
   - Goal: after a hit already exists, infer who hit whom.
   - Authority: this path enriches a hit with source/trajectory attribution.
   - It must not become a required gate for final hit.

Global bullet reconstruction is evidence enrichment, not hit existence evidence.
If reconstruction is required before final hit, missed reconstruction will reintroduce missed-hit risk.

## Contract Boundary

`hit_judge_server.py` remains the authority for hit candidates.

Allowed final-hit evidence:
- event bullet point or stable event bullet track contact
- terminal point inside human mask
- turning point inside human mask
- mask/person sync evidence
- trigger-period/sync evidence
- short local approach continuity
- local anti-noise checks against human-motion event fragments

Explicitly not required:
- global bullet track reconstructed across cameras
- shooter/source attribution
- global BEV track continuity
- speed/trajectory attribution service result

## Target 1: Hit Candidate Optimization

Inputs:
- `overlay_bullet_point_*.jsonl`
- `overlay_bullet_point_all.jsonl`
- per-camera person/mask evidence
- trigger/sync evidence
- event worker bullet IDs / stable track IDs

Outputs:
- `hit_candidate_{camera}.jsonl`
- `hit_candidate_all.jsonl`
- hit judge status/debug streams

V20 candidate fields should make the split explicit:
- `hit_candidate_id`
- `hit_authority: "hit_judge"`
- `candidate_kind: "human_hit_candidate"`
- `hit_decision_stage: "candidate" | "final"`
- `local_bullet_id`
- `event_stable_track_id`
- `person_stable_id`
- `evidence_mask_ok`
- `evidence_trigger_ok`
- `evidence_person_sync_ok`
- `noise_reject_reason`
- `attribution_required: false`

Optimization direction:
- Prefer recall for real terminal/turning hits.
- Keep local evidence gating, but avoid requiring long global trajectories.
- Use the event worker stable track ID as the local trajectory identity source.
- Preserve debug rows for rejected noise, especially human-motion fragments.

## Target 2: Global Bullet Reconstruction And Damage Attribution

This path runs as an independent sidecar.

Inputs:
- final/candidate hit stream from `hit_candidate_all.jsonl`
- global bullet tracks from event bullet fusion
- global person tracks
- optional foam dart speed diagnostics

Outputs:
- `damage_attribution_candidates.jsonl`
- `damage_attribution_latest.json`
- `damage_attribution_status.json`
- optional game event bridge output

Attribution fields:
- `hit_candidate_id`
- `victim_global_player_id`
- `source_global_player_id`
- `source_score`
- `source_score_terms`
- `global_bullet_track_id`
- `trajectory_reconstruction_state`
- `attribution_state: "resolved" | "ambiguous" | "unresolved"`

Important:
- A hit can be valid even when attribution is unresolved.
- Attribution may update later as more global trajectory context arrives.
- Downstream game logic can choose whether unresolved attribution has a default rule, but hit existence should not be rolled back.

## Runtime Relationship

```text
Event worker / MVS / YOLO
        |
        v
Hit Judge
  - detects hit candidates
  - emits final hit candidates
  - does not wait for global reconstruction
        |
        +----------------------+
        |                      |
        v                      v
Operator / game hit event      Global bullet reconstruction
                               Damage attribution
                               Game source inference
```

## Development Order

1. Add schema fields to hit candidates to make authority and attribution separation explicit.
2. Audit `hit_judge_server.py` for any final-hit condition that implicitly depends on global trajectory or attribution.
3. Improve hit candidate recall/anti-noise rules locally:
   - terminal point in mask
   - turning point in mask
   - stable event track continuity
   - local human-motion noise rejection
4. Keep global bullet fusion and damage attribution as sidecar/shadow first.
5. Add replay comparison metrics:
   - real hit recall
   - human-motion false positive count
   - unresolved attribution count
   - resolved attribution accuracy when ground truth exists

## Acceptance Criteria

Hit candidate path:
- Real human-hit samples should not require global trajectory reconstruction to emit a hit.
- Human-motion event noise should be diagnosable and reducible through local evidence rules.
- Every rejected candidate should have a reason suitable for dataset analysis.

Attribution path:
- Can attach source/victim confidence after hit exists.
- Can be unresolved without invalidating the hit.
- Does not publish back into `hit_judge_server.py` as a final-hit gate.

## Known Risk

Separating hit and attribution means some hits will initially have no shooter/source.
This is intentional for recall. The game layer must tolerate unresolved attribution or apply a separate fallback policy.
