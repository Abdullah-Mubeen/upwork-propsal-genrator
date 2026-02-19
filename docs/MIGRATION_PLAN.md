# Multi-Tenant Migration Plan: MongoDB + Pinecone

## Executive Summary

This document outlines a **zero-downtime, zero-data-loss** migration strategy for transitioning from single-tenant to multi-tenant architecture. The migration involves:

- **MongoDB**: Migrate `training_data` → `portfolio_items` with new lean schema
- **Pinecone**: Cleanup 612 legacy vectors → Re-embed 101 vectors with namespace isolation
- **New Collections**: `organizations`, `freelancer_profiles` with proper relationships

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Target State Architecture](#target-state-architecture)
3. [Risk Assessment](#risk-assessment)
4. [Migration Phases](#migration-phases)
5. [Backup Strategy](#backup-strategy)
6. [Rollback Procedures](#rollback-procedures)
7. [Validation Checklist](#validation-checklist)
8. [Execution Timeline](#execution-timeline)

---

## Current State Analysis

### MongoDB Collections

| Collection | Records | Description |
|------------|---------|-------------|
| `training_data` | 101 | Old 5-chunk schema with job data |
| `chunks` | ~500 | Chunked training data |
| `embeddings` | ~500 | Embedding metadata |
| `portfolio_items` | 0 | New lean schema (target) |
| `organizations` | 0 | Multi-tenant orgs (new) |
| `freelancer_profiles` | 0 | Profile data (new) |

### Pinecone Index

| Namespace | Vectors | Strategy |
|-----------|---------|----------|
| `proposals` | 612 | Legacy 5-chunk per project |
| Target | 101 | 1 vector per portfolio item |

### Data Relationships (Current)
```
training_data (standalone) ─┬─> chunks ─> embeddings ─> Pinecone (flat namespace)
```

### Data Relationships (Target)
```
organizations
    └─> freelancer_profiles
            └─> portfolio_items ─> Pinecone (org-scoped namespace)
```

---

## Target State Architecture

### New Schema: portfolio_items (5-field lean schema)

```json
{
    "item_id": "port_xxx",
    "org_id": "org_xxx",
    "profile_id": "prof_xxx",
    "company_name": "Acme Corp",
    "deliverables": ["Performance audit", "Core Web Vitals fixes"],
    "skills": ["React", "Node.js", "PostgreSQL"],
    "portfolio_url": "https://example.com/project",
    "industry": "E-commerce",
    "is_embedded": true,
    "embedding_id": "port_xxx"
}
```

### Pinecone Namespace Strategy

```
Before: proposals (flat, 612 vectors)
After:  portfolio_default      (for migration - default org)
        portfolio_acme-inc     (future orgs)
        portfolio_agency-xyz   (future orgs)
```

---

## Risk Assessment

### High-Impact Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data loss during migration | Critical | Full backups + validation checksums |
| Pinecone quota exceeded | High | Batch processing + rate limiting |
| Schema mismatch | High | Schema validation before insert |
| Embedding API failures | Medium | Retry logic + checkpoint system |
| Downtime during migration | Medium | Blue-green approach |

### Data Integrity Guarantees

1. **Pre-migration backup**: Full MongoDB dump + Pinecone vector export
2. **Transactional migration**: Each record validated before commit
3. **Checksum verification**: MD5 hash comparison pre/post migration
4. **Rollback checkpoint**: Restore point after each phase

---

## Migration Phases

### Phase 0: Preparation (Pre-flight)
```bash
# 1. Take full backups
python scripts/backup/mongodb_backup.py
python scripts/backup/pinecone_backup.py

# 2. Verify backup integrity
python scripts/validate_migration.py --phase=backup
```

### Phase 1a: Create Default Organization
```python
# Creates: org_default with slug "default-org"
{
    "org_id": "org_default",
    "name": "Default Organization",
    "slug": "default-org",
    "org_type": "individual",
    "plan_tier": "pro"  # Allows migration without limits
}
```

### Phase 1b: Create Default Profile
```python
# Creates: prof_default linked to org_default
{
    "profile_id": "prof_default",
    "org_id": "org_default",
    "name": "Default Freelancer",
    "title": "Full-Stack Developer",
    ...
}
```

### Phase 2: Migrate Portfolio Data
```bash
# Migrate 101 records from training_data → portfolio_items
python scripts/migrate_training_to_portfolio_v2.py \
    --org_id=org_default \
    --profile_id=prof_default \
    --dry-run  # Preview first

# Then execute:
python scripts/migrate_training_to_portfolio_v2.py \
    --org_id=org_default \
    --profile_id=prof_default
```

### Phase 3: Pinecone Re-indexing
```bash
# 1. Clear legacy vectors (optional backup first)
python scripts/cleanup_pinecone_reembed.py --action=backup

# 2. Re-embed with new metadata
python scripts/cleanup_pinecone_reembed.py \
    --action=reembed \
    --namespace=portfolio_default-org
```

### Phase 4: Validation & Cutover
```bash
# Validate all migrations
python scripts/validate_migration.py --phase=all

# Switch application config
# Update PINECONE_NAMESPACE in .env to "portfolio_default-org"
```

---

## Backup Strategy

### MongoDB Backup

```bash
# Full export to JSON (portable)
python scripts/backup/mongodb_backup.py --output=/backups/pre_migration

# Creates:
# /backups/pre_migration/
#   ├── training_data.json      (101 records)
#   ├── chunks.json             (~500 records)
#   ├── embeddings.json         (~500 records)
#   ├── proposals.json
#   └── manifest.json           (checksums + metadata)
```

### Pinecone Backup

```bash
# Export all vectors with metadata
python scripts/backup/pinecone_backup.py --output=/backups/pinecone

# Creates:
# /backups/pinecone/
#   ├── vectors_proposals.jsonl  (612 vectors)
#   └── manifest.json            (namespace stats)
```

### Restore Commands

```bash
# MongoDB restore
python scripts/backup/mongodb_backup.py --restore=/backups/pre_migration

# Pinecone restore
python scripts/backup/pinecone_backup.py --restore=/backups/pinecone
```

---

## Rollback Procedures

### Scenario 1: Migration Script Failure

```bash
# 1. Stop migration
# 2. Delete partially migrated data
python scripts/run_migration.py --rollback=phase2

# 3. Restore from backup if needed
python scripts/backup/mongodb_backup.py --restore=/backups/pre_migration
```

### Scenario 2: Pinecone Re-index Failure

```bash
# 1. Clear corrupted namespace
python scripts/cleanup_pinecone_reembed.py --action=clear --namespace=portfolio_default-org

# 2. Restore from backup
python scripts/backup/pinecone_backup.py --restore=/backups/pinecone

# 3. Retry re-indexing
python scripts/cleanup_pinecone_reembed.py --action=reembed --namespace=portfolio_default-org
```

### Scenario 3: Full Rollback (Abort Migration)

```bash
# 1. Restore all MongoDB collections
python scripts/backup/mongodb_backup.py --restore=/backups/pre_migration

# 2. Restore Pinecone vectors
python scripts/backup/pinecone_backup.py --restore=/backups/pinecone

# 3. Revert config changes
# Update .env PINECONE_NAMESPACE back to "proposals"

# 4. Delete new collections (optional)
python scripts/run_migration.py --rollback=all
```

---

## Validation Checklist

### Pre-Migration Checks

- [ ] Backup completed successfully
- [ ] Backup manifest checksums verified
- [ ] Application stopped or in maintenance mode
- [ ] Database connection tested
- [ ] Pinecone API key validated
- [ ] Sufficient Pinecone quota available

### Post-Migration Checks

| Check | Command | Expected |
|-------|---------|----------|
| Portfolio count | `db.portfolio_items.countDocuments()` | 101 |
| Embedded count | `db.portfolio_items.countDocuments({is_embedded: true})` | 101 |
| Pinecone vectors | `index.describe_index_stats()` | 101 in new namespace |
| Org created | `db.organizations.findOne({org_id: "org_default"})` | Not null |
| Profile created | `db.freelancer_profiles.findOne({profile_id: "prof_default"})` | Not null |
| No orphaned vectors | `validate_migration.py --check=orphans` | 0 orphans |
| Retrieval works | `test_proposal_generation.py` | Proposals generated |

---

## Execution Timeline

### Estimated Duration: ~2 hours (with validation)

| Phase | Duration | Description |
|-------|----------|-------------|
| Phase 0 | 15 min | Backups + verification |
| Phase 1a | 2 min | Create default org |
| Phase 1b | 2 min | Create default profile |
| Phase 2 | 30 min | Migrate 101 records |
| Phase 3 | 45 min | Re-embed to Pinecone (rate limited) |
| Phase 4 | 15 min | Validation + cutover |
| Buffer | 15 min | Issues + retries |

### Recommended Schedule

1. **T-1 day**: Take backups, test restore procedure
2. **T-0 hour**: Put app in maintenance mode (optional)
3. **T+0**: Execute migration phases 1-4
4. **T+2 hours**: Validation complete, resume normal operation
5. **T+24 hours**: Remove legacy collections (post-validation)

---

## Commands Quick Reference

```bash
# Full migration (recommended)
python scripts/run_migration.py --all

# Step-by-step
python scripts/run_migration.py --phase=0  # Backup
python scripts/run_migration.py --phase=1  # Create org/profile
python scripts/run_migration.py --phase=2  # Migrate data
python scripts/run_migration.py --phase=3  # Pinecone re-index
python scripts/run_migration.py --phase=4  # Validate

# Dry run (preview all changes)
python scripts/run_migration.py --all --dry-run

# Rollback
python scripts/run_migration.py --rollback=phase2  # Specific phase
python scripts/run_migration.py --rollback=all     # Full rollback
```

---

## Support Contacts

- **MongoDB Issues**: Check connection string, verify credentials
- **Pinecone Issues**: Check API key, verify quota at console.pinecone.io
- **Application Issues**: Check logs in `/var/log/proposal_generator/`

---

## Appendix: Field Mapping Reference

### training_data → portfolio_items (5-field lean schema)

| Old Field | New Field | Transformation |
|-----------|-----------|----------------|
| `company_name` or `job_title` | `company_name` | First non-empty |
| `skills_required` | `skills` | Top 10 |
| `job_title` + `task_type` | `deliverables` | Extracted |
| `portfolio_urls[0]` | `portfolio_url` | First URL |
| `industry` | `industry` | Direct copy or extracted |
| New | `org_id` | Default org ID |
| New | `profile_id` | Default profile ID |

**Removed Fields (previously in 8-field schema):**
- `outcome` - Removed
- `client_feedback` - Removed
- `duration_days` - Removed
