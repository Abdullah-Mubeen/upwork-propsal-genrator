# Backend Architecture Audit Report

**Date:** February 2026  
**Auditor:** AI Solution Architect  
**Project:** Upwork Proposal Generator  

---

## 🔄 CLEANUP UPDATE (Latest)

**Date:** June 2025

Major cleanup completed as part of GitHub Issue #31:

| Item | Before | After | Status |
|------|--------|-------|--------|
| `db.py` | 1,390 lines | ~366 lines (74% reduction) | ✅ Repository shims |
| `advanced_chunker.py` | 619 lines | DELETED (stub in job_data_processor.py) | ✅ |
| `data_chunker.py` | 120 lines | DELETED | ✅ |
| `proposal_generator.py` | 728 lines | DELETED | ✅ |
| `ImportSource` enum | 2 definitions | 1 in constants.py | ✅ |

**New Architecture:**
- Repository pattern in `app/infra/mongodb/repositories/`
- Centralized constants in `app/domain/constants.py`
- Legacy files marked with DEPRECATED warnings

---

## Executive Summary

The codebase has grown organically through reactive feature additions, resulting in:
- **~12,000+ lines of Python** in the backend
- **Significant code duplication** across modules
- **728 lines of unused production code** (`proposal_generator.py`) → ✅ DELETED
- **Single-tenant architecture** hardcoded
- **No clear domain boundaries**

This audit identifies cleanup targets and provides a redesign path for multi-tenant support.

---

## Code Size Analysis

| File | Lines | Status |
|------|-------|--------|
| `db.py` | ~~1,390~~ 366 | ✅ REDUCED - uses repository shims |
| `prompt_engine.py` | 1,347 | ⚠️ Over-engineered |
| `job_data_processor.py` | 1,111 | ⚠️ LEGACY - marked deprecated |
| `hook_strategy.py` | 946 | ⚠️ Could simplify |
| `job_data_ingestion.py` | 929 | ⚠️ LEGACY - marked deprecated |
| `metadata_extractor.py` | 910 | ⚠️ Duplicates logic |
| `retrieval_pipeline.py` | 882 | Core retrieval |
| `openai_service.py` | 875 | External API wrapper |
| `proposals.py` | 837 | Route file |
| `proposal_generator.py` | ~~728~~ | ✅ **DELETED** |
| `advanced_chunker.py` | ~~619~~ | ✅ **DELETED** |
| `pinecone_service.py` | 458 | Vector DB service |
| `job_data_schema.py` | 472 | Pydantic models |

**Total: ~12,470 lines**

---

## Critical Issues Found

### 1. Dead/Unused Code ❌ → ✅ RESOLVED

#### `proposal_generator.py` (728 lines) - ~~COMPLETELY UNUSED~~ **DELETED**

**Status:** ✅ Already deleted - file no longer exists.

**Why it was deleted:**
- Never imported in production routes (`app/routes/proposals.py`)
- Only imported in test files (which have been fixed to skip the deprecated tests)
- All proposal generation is done inline in `proposals.py` using `RetrievalPipeline + PromptEngine + OpenAIService`
- 728 lines of code that duplicated existing functionality

**Test files updated:**
- `test_proposal_generation.py` - removed dead import
- `test_geration_retrieval.py` - added pytest.skip to tests using deleted class

---

### 2. Code Duplication 🔄 → ✅ RESOLVED

**Status:** ✅ Consolidated into `app/domain/constants.py`

All duplicate constants have been moved to a single source of truth:
- `AI_ML_KEYWORDS` - Merged from retrieval_pipeline.py + hook_strategy.py
- `PLATFORM_KEYWORDS` - Moved from retrieval_pipeline.py
- `PAIN_POINT_INDICATORS` - Moved from prompt_engine.py
- `URGENCY_PATTERNS` - Moved from hook_strategy.py
- `URGENCY_TIMELINE_PROMISES` - Moved from prompt_engine.py
- `EMPATHY_RESPONSES` - Moved from prompt_engine.py
- `INDUSTRY_KEYWORDS` - Moved from metadata_extractor.py
- `BRAND_INDUSTRY_MAP` - Moved from metadata_extractor.py
- `COMPLEXITY_INDICATORS` - Moved from metadata_extractor.py
- `CLIENT_INTENT_KEYWORDS` - Moved from metadata_extractor.py

**Files updated:**
- `retrieval_pipeline.py` - Now imports from constants
- `hook_strategy.py` - Now imports from constants
- `prompt_engine.py` - Now imports from constants
- `metadata_extractor.py` - Now imports from constants

**Net reduction:** ~330 lines of duplicate code removed

---

### ~~3.~~ Over-Engineered Components 🔧 → Partially Resolved

#### A. Chunking System Evolution → ✅ RESOLVED

The chunking went through 3 generations:
1. **v1:** Basic chunking (now deprecated methods in `data_chunker.py`)
2. **v2:** 4-chunk strategy (transitional)
3. **v3:** 5-layer semantic chunking (`advanced_chunker.py`) → ✅ DELETED

**Status:** ✅ Both `data_chunker.py` AND `advanced_chunker.py` DELETED

The chunking strategy went through multiple iterations. The current state:
- `advanced_chunker.py` - DELETED (was 619 lines)
- `data_chunker.py` - DELETED (120-line wrapper)
- `job_data_processor.py` - Contains stub class that raises DeprecationWarning
- New code should use `app/services/job_ingestion_service.py`

**Files updated:**
- `job_data_ingestion.py` - Marked as LEGACY with deprecation notice
- `job_data_processor.py` - Marked as LEGACY, has stub AdvancedChunkProcessor

**Impact:** -121 lines, cleaner architecture

---

#### B. `db.py` is a God Object (1,390 lines) - ✅ RESOLVED

**Status:** ✅ Reduced from 1,390 → 366 lines (74% reduction)

**Solution implemented:**
- Created repository pattern in `app/infra/mongodb/repositories/`
- `db.py` now contains thin shims that delegate to repositories
- Each repository handles a single domain concern

**Repository structure:**
```
app/infra/mongodb/repositories/
├── training_repo.py      # Training data, chunks, embeddings
├── proposal_repo.py      # Proposals, sent proposals, feedback
├── analytics_repo.py     # Analytics collection
├── profile_repo.py       # User profiles
├── admin_repo.py         # Admin API keys
├── org_repo.py           # Multi-tenant organizations
├── user_repo.py          # Multi-tenant users
├── portfolio_repo.py     # Portfolio entries
└── job_prefs_repo.py     # Job preferences/filters
```

**Impact:** Clean separation of concerns, easier testing, ready for multi-tenant

---

~~Should be split into:~~
~~- `db/connection.py` - Connection management~~
~~- `repositories/training_data.py`~~
~~- `repositories/proposals.py`~~
~~- `repositories/analytics.py`~~
~~- `repositories/admin.py`~~

---

#### C. `prompt_engine.py` (1,347 lines)

Contains:
- Style instructions (~100 lines)
- Tone instructions (~50 lines)
- Pain point indicators (~100 lines)
- Urgency detection (~100 lines)
- Empathy responses (~100 lines)
- Hook generation (~200 lines) - duplicates `hook_strategy.py`
- Prompt building (~400 lines)
- Quality scoring (~150 lines)
- Constants that should be in config

---

### 4. Single-Tenant Hardcoding 🔒

#### Current State

```python
# db.py line 1030
profile = self.db["user_profile"].find_one({"user_id": "default"})

# db.py line 1034
"user_id": "default",
```

**ALL data is global** - no isolation between:
- Users
- Organizations/Agencies
- API keys (only track role, not ownership)

#### Collections Affected

| Collection | Has user_id? | Has org_id? |
|------------|--------------|-------------|
| `training_data` | ❌ | ❌ |
| `chunks` | ❌ | ❌ |
| `embeddings` | ❌ | ❌ |
| `proposals` | ❌ | ❌ |
| `sent_proposals` | ❌ | ❌ |
| `feedback_data` | ❌ | ❌ |
| `skills` | ❌ | ❌ |
| `user_profile` | ✅ (hardcoded "default") | ❌ |
| `api_keys` | ❌ (only role) | ❌ |

---

## Current Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CURRENT ARCHITECTURE (MESSY)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ROUTES LAYER (Thin? NO - Fat controllers)                                  │
│  ├── proposals.py (837 lines) ────────────────────────────────────────────┐ │
│  │     Contains business logic inline instead of calling services         │ │
│  │     Duplicates what proposal_generator.py does                         │ │
│  │                                                                        │ │
│  └── job_data_ingestion.py (911 lines) ──────────────────────────────────┐│ │
│        Also contains business logic                                       ││ │
│                                                                          ││ │
│  UTILS LAYER (Reduced - several files deleted)                           ││ │
│  ├── prompt_engine.py (1347) ←──┐                                        ││ │
│  │                              │ Duplicate pain point detection         ││ │
│  ├── hook_strategy.py (946) ←───┘                                        ││ │
│  │                                                                       ││ │
│  ├── metadata_extractor.py (910) ←──┐                                    ││ │
│  │                                  │ 3 different industry detections    ││ │
│  ├── openai_service.py (875) ←──────┘                                    ││ │
│  │                                                                       ││ │
│  ├── proposal_generator.py ← ✅ DELETED                                  ││ │
│  │                                                                       ││ │
│  ├── retrieval_pipeline.py (882) ← Duplicates AI_ML_KEYWORDS             ││ │
│  │                                                                       ││ │
│  ├── data_chunker.py ← ✅ DELETED                                        ││ │
│  │     └── advanced_chunker.py ← ✅ DELETED                              ││ │
│  │                                                                       ││ │
│  └── job_data_processor.py (1111) ← LEGACY, marked deprecated            ││ │
│                                                                          ││ │
│  DB LAYER (REFACTORED)                                                   ││ │
│  └── db.py (366 lines) ← Now uses repository shims                       ││ │
│        └── app/infra/mongodb/repositories/ ← New repository layer        ││ │
│                                                                           │ │
│  NO SEPARATION OF CONCERNS                                                │ │
└───────────────────────────────────────────────────────────────────────────┘ │
```

---

## MongoDB Collections Audit

### Current Collections (13)

| # | Collection | Purpose | Records (approx) |
|---|------------|---------|------------------|
| 1 | `training_data` | Raw job data | Main data |
| 2 | `chunks` | Semantic chunks | 5 per job |
| 3 | `embeddings` | Embedding metadata | 1 per chunk |
| 4 | `proposals` | Historical proposals | Few |
| 5 | `sent_proposals` | Outcome tracking | User input |
| 6 | `feedback_data` | Client feedback | From images/text |
| 7 | `skills` | Skill frequency | Unique skills |
| 8 | `skill_embeddings` | Skill vectors | Unique skills |
| 9 | `embedding_cache` | Text→embedding cache | Performance |
| 10 | `user_profile` | User settings | **1 (singleton!)** |
| 11 | `api_keys` | Auth keys | Few |
| 12 | `activity_log` | Audit trail | Many |
| 13 | (implicit) | `jobs` collection? | Check if exists |

### Redundancy Analysis

- `chunks` + `embeddings` could be merged (embedding stored on chunk)
- `skills` + `skill_embeddings` could be merged
- `embedding_cache` is useful but rarely hit

---

## Pinecone Usage Analysis

### Current State

- **Index:** `proposal-engine`
- **Dimension:** 3072 (text-embedding-3-large)
- **Namespace:** `proposals` (single namespace for ALL data)

### Problem

All vectors are in ONE namespace. For multi-tenant:
- Option A: Namespace per org (100 namespace limit)
- Option B: Metadata filter by `org_id`
- Option C: Hybrid

---

## Cleanup Targets (Priority Order)

### P0 - DELETE (No Risk)

| File | Lines | Action |
|------|-------|--------|
| `proposal_generator.py` | 728 | DELETE - unused in production |

### P1 - CONSOLIDATE (Medium Risk)

| Target | From | To | Lines Saved |
|--------|------|----|-------------|
| AI_ML_KEYWORDS | 2 files | `constants.py` | ~30 |
| INDUSTRY_KEYWORDS | 1 file | `constants.py` | ~50 |
| PLATFORM_KEYWORDS | 1 file | `constants.py` | ~30 |
| Pain point detection | 2 files | Single util | ~80 |
| Industry detection | 3 files | Single service | ~100 |
| Urgency detection | 2 files | Single util | ~50 |

### P2 - SIMPLIFY (Higher Risk)

| Target | Current | Proposed | Effort |
|--------|---------|----------|--------|
| `data_chunker.py` | ~~Wrapper~~ | ✅ DELETED | Done |
| `advanced_chunker.py` | ~~619 lines~~ | ✅ DELETED | Done |
| `db.py` | ~~God object~~ | ✅ Repository shims (366 lines) | Done |
| `prompt_engine.py` | 1347 lines | Split into focused modules | High |

---

## Next Steps

1. **CREATE** `docs/ARCHITECTURE_REDESIGN.md` - Target architecture
2. **CREATE** GitHub issues for each cleanup task
3. **PRIORITIZE** sprints based on risk/impact
4. **IMPLEMENT** foundation before adding features

---

## Files Created by This Audit

- `docs/ARCHITECTURE_AUDIT.md` (this file)
- `docs/ARCHITECTURE_REDESIGN.md` (next)
- `docs/GITHUB_ISSUES.md` (sprint planning)
