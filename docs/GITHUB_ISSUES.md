# GitHub Issues & Sprint Planning

This document contains ready-to-create GitHub issues organized by sprint and priority.

---

## Labels to Create

```
Priority:
- P0-critical
- P1-high  
- P2-medium
- P3-low

Type:
- cleanup
- feature
- refactor
- bug
- docs
- test

Area:
- backend
- database
- auth
- frontend
- infra
```

---

## Epic: Technical Debt Cleanup

### Milestone: Sprint 1 - Code Cleanup (Week 1)

---

#### Issue #1: Delete unused `proposal_generator.py` ✅ COMPLETED

**Labels:** `P0-critical`, `cleanup`, `backend`

**Status:** ✅ Completed in commit `cleanup: remove dead code, fix tests, add architecture docs`

**Description:**
The file `app/utils/proposal_generator.py` (728 lines) was **completely unused in production**. It was only imported in test files, but the actual proposal generation happens inline in `proposals.py`.

**Tasks:**
- [x] Verify no production code depends on this file
- [x] Update tests to use `proposals.py` approach or mock
- [x] Delete `app/utils/proposal_generator.py`
- [x] Remove any dangling imports

**Impact:** -728 lines of dead code

---

#### Issue #2: Create centralized constants file ✅ COMPLETED

**Labels:** `P1-high`, `refactor`, `backend`

**Status:** ✅ Completed in commit `consolidate: centralize constants, remove data_chunker wrapper`

**Description:**
Keywords and constants were duplicated across multiple files. Created `app/domain/constants.py` as single source of truth.

**Constants consolidated:**
- `AI_ML_KEYWORDS` - From retrieval_pipeline.py + hook_strategy.py
- `PLATFORM_KEYWORDS` - From retrieval_pipeline.py
- `PAIN_POINT_INDICATORS` - From prompt_engine.py
- `URGENCY_PATTERNS` - From hook_strategy.py
- `URGENCY_TIMELINE_PROMISES` - From prompt_engine.py
- `EMPATHY_RESPONSES` - From prompt_engine.py
- `INDUSTRY_KEYWORDS` - From metadata_extractor.py
- `BRAND_INDUSTRY_MAP` - From metadata_extractor.py
- `COMPLEXITY_INDICATORS` - From metadata_extractor.py
- `CLIENT_INTENT_KEYWORDS` - From metadata_extractor.py

**Tasks:**
- [x] Create `app/domain/constants.py`
- [x] Move ALL keyword constants to single file
- [x] Update all imports across codebase
- [ ] Add tests for constant lookups (deferred)

**Impact:** ~330 lines of duplicate code removed

---

#### Issue #3: Consolidate text analysis utilities ✅ COMPLETED

**Labels:** `P1-high`, `refactor`, `backend`

**Status:** ✅ Completed in commit `consolidate: create text_analysis.py, finish sprint 1`

**Description:**
Pain point detection and urgency detection were implemented multiple times. Now consolidated into `app/utils/text_analysis.py`.

**Functions consolidated:**
- `detect_urgency()` - Returns UrgencyResult with level (1-5) AND label
- `detect_urgency_level()` - Legacy wrapper returning string label
- `detect_urgency_score()` - Legacy wrapper returning int level
- `extract_pain_points()` - Returns Dict[str, List[str]] with categories
- `extract_pain_points_simple()` - Legacy wrapper returning flat list
- `extract_specific_details()` - Extract numbers, tools, metrics
- `extract_tone_words()` - Extract emotional/tone words
- `analyze_job_text()` - Comprehensive analysis combining all functions

**Tasks:**
- [x] Create `app/utils/text_analysis.py`
- [x] Implement single `detect_pain_points()` function
- [x] Implement single `detect_urgency()` function
- [x] Update prompt_engine.py to delegate to text_analysis
- [x] Update hook_strategy.py to delegate to text_analysis
- [ ] Implement single `detect_industry()` function (deferred - already consolidated in metadata_extractor)
- [ ] Add unit tests (deferred)

**Impact:** ~150 lines of duplicate code consolidated

---

#### Issue #4: Remove `data_chunker.py` wrapper ✅ COMPLETED

**Labels:** `P2-medium`, `cleanup`, `backend`

**Status:** ✅ Completed in commit `consolidate: centralize constants, remove data_chunker wrapper`

**Description:**
`data_chunker.py` was a 120-line wrapper that just delegated to `advanced_chunker.py`.

**Before:**
```
job_data_ingestion.py → DataChunker → AdvancedChunkProcessor
```

**After:**
```
job_data_ingestion.py → AdvancedChunkProcessor (direct)
```

**Tasks:**
- [x] Add `chunk_training_data` alias to `AdvancedChunkProcessor`
- [x] Update imports in `job_data_ingestion.py`
- [x] Update imports in `job_data_processor.py`
- [x] Delete `app/utils/data_chunker.py`
- [x] Verify tests pass

**Impact:** -121 lines, cleaner import path

**Impact:** -120 lines, cleaner import path

---

### Milestone: Sprint 2 - Restructure (Week 2)

---

#### Issue #5: Split `db.py` into repositories

**Labels:** `P1-high`, `refactor`, `backend`, `database`

**Description:**
`db.py` is a 1,390-line "god object" handling:
- Connection management
- CRUD for 13 collections
- Analytics queries
- Caching logic
- Admin functions

**Target structure:**
```
app/infra/mongodb/
├── connection.py              # Connection management (~50 lines)
├── base_repository.py         # Generic CRUD (~100 lines)
└── repositories/
    ├── training_repo.py       # training_data, chunks, embeddings
    ├── proposal_repo.py       # proposals, sent_proposals
    ├── feedback_repo.py       # feedback_data
    ├── analytics_repo.py      # Statistics queries
    └── admin_repo.py          # api_keys, activity_log
```

**Tasks:**
- [ ] Create `app/infra/mongodb/` directory structure
- [ ] Extract `DatabaseManager.__init__` to `connection.py`
- [ ] Create `BaseRepository` with common CRUD
- [ ] Migrate training data methods to `training_repo.py`
- [ ] Migrate proposal methods to `proposal_repo.py`
- [ ] Migrate analytics methods to `analytics_repo.py`
- [ ] Update all imports
- [ ] Test each repository independently

**Impact:** Better separation of concerns, easier testing

---

#### Issue #6: Create service layer

**Labels:** `P1-high`, `refactor`, `backend`

**Description:**
Business logic is currently embedded in route handlers (`proposals.py` has 837 lines). Extract into focused services.

**Target structure:**
```
app/services/
├── proposal_service.py        # Proposal generation orchestration
├── training_service.py        # Training data pipeline
├── retrieval_service.py       # Vector search + ranking
├── embedding_service.py       # OpenAI embedding calls
└── analytics_service.py       # Stats computation
```

**Tasks:**
- [ ] Create `app/services/` directory
- [ ] Extract proposal generation from `proposals.py` → `proposal_service.py`
- [ ] Extract training pipeline from routes → `training_service.py`
- [ ] Keep route files thin (just validation + service calls)
- [ ] Add service-level tests

---

#### Issue #7: Simplify `prompt_engine.py`

**Labels:** `P2-medium`, `refactor`, `backend`

**Description:**
`prompt_engine.py` is 1,347 lines containing:
- Style/tone instructions (should be config/templates)
- Pain point detection (duplicates hook_strategy)
- Empathy responses (could be templates)
- Hook generation (duplicates hook_strategy)
- Prompt building
- Quality scoring

**Tasks:**
- [ ] Move style/tone templates to `app/templates/prompts/`
- [ ] Use consolidated `text_analysis.py` for detection
- [ ] Merge hook generation with `hook_strategy.py`
- [ ] Create focused `PromptBuilder` class (~300 lines max)

---

### Milestone: Sprint 3 - Multi-Tenant Foundation (Week 3)

---

#### Issue #8: Create `organizations` collection

**Labels:** `P0-critical`, `feature`, `database`

**Description:**
Add organization support for multi-tenant architecture.

**Schema:**
```javascript
{
  "org_id": "org_abc123",
  "name": "Acme Agency",
  "slug": "acme-agency",
  "type": "agency",              // agency | individual
  "owner_id": "user_xyz",
  "settings": { ... },
  "limits": { ... },
  "pinecone_config": {
    "namespace": "org_abc123"
  },
  "billing": { ... },
  "created_at": ISODate,
  "is_active": true
}
```

**Tasks:**
- [ ] Create organization Pydantic models
- [ ] Add `organizations` collection with indexes
- [ ] Create `organization_repo.py`
- [ ] Create `organization_service.py` with CRUD
- [ ] Create `/api/v1/organizations.py` routes
- [ ] Add tests

---

#### Issue #9: Create `users` collection (replace `user_profile`)

**Labels:** `P0-critical`, `feature`, `database`

**Description:**
Replace singleton `user_profile` with proper multi-user support.

**Schema:**
```javascript
{
  "user_id": "user_xyz789",
  "org_id": "org_abc123",        // FK to organizations
  "email": "john@acme.com",
  "name": "John Doe",
  
  // Role depends on org.type:
  // - For agency orgs: "owner" | "admin" | "member"  
  // - For individual orgs: always "owner" (they are their own admin)
  "role": "admin",
  
  "permissions": [...],          // Derived from role
  
  "profile": {
    "upwork_url": "https://upwork.com/...",
    "bio": "Full-stack developer...",
    "hourly_rate": 75,
    "skills": ["React", "Node.js"],
    "proposal_template_preferences": {
      "default_style": "conversational",
      "include_portfolio": true
    }
  },
  
  "api_key_hash": "...",
  "created_at": ISODate,
  "last_login": ISODate
}

// NOTE: For Individual (Self-Managed) accounts:
// - org.type = "individual"
// - user.role = "owner" (always)
// - They manage their own billing
// - Not associated with any agency
```

**Tasks:**
- [ ] Create user Pydantic models
- [ ] Add `users` collection with indexes
- [ ] Migrate existing `user_profile` data
- [ ] Create `user_repo.py`
- [ ] Create `/api/v1/users.py` routes
- [ ] Update auth middleware to use users
- [ ] Handle Individual vs Agency registration flows

---

#### Issue #10: Add `org_id` to all collections

**Labels:** `P0-critical`, `refactor`, `database`

**Description:**
All data must be tenant-isolated. Add `org_id` to:
- `training_data`
- `chunks`
- `embeddings`
- `proposals`
- `sent_proposals`
- `feedback_data`
- `skills`
- `skill_embeddings`

**Tasks:**
- [ ] Create migration script
- [ ] Add `org_id` field to all collections
- [ ] Create indexes on `org_id`
- [ ] Update all repository queries to filter by `org_id`
- [ ] Update Pinecone upserts to include `org_id` metadata
- [ ] Test isolation between orgs

---

#### Issue #11: Create tenant middleware

**Labels:** `P1-high`, `feature`, `backend`, `auth`

**Description:**
Inject tenant context into every request based on authenticated user.

```python
# Example middleware
@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    user = get_current_user(request)
    request.state.org_id = user.org_id
    request.state.user_id = user.user_id
    response = await call_next(request)
    return response
```

**Tasks:**
- [ ] Create `app/api/middleware/tenant.py`
- [ ] Extract org_id from authenticated user
- [ ] Inject into request state
- [ ] Update services to use tenant context
- [ ] Add tests for tenant isolation

---

### Milestone: Sprint 4 - Auth & RBAC (Week 4)

---

#### Issue #12: Implement JWT authentication

**Labels:** `P1-high`, `feature`, `auth`

**Description:**
Replace simple API key auth with proper JWT for user sessions.

**Tasks:**
- [ ] Add `python-jose` for JWT
- [ ] Create `/auth/login` endpoint
- [ ] Create `/auth/register` endpoint
- [ ] Create `/auth/refresh` endpoint
- [ ] Update auth middleware to validate JWT
- [ ] Keep API key support for programmatic access
- [ ] Add rate limiting per user

---

#### Issue #13: Implement organization registration flow

**Labels:** `P1-high`, `feature`, `backend`

**Description:**
Allow new organizations to register.

**Flow:**
1. User signs up → Creates individual org by default
2. User creates agency → New org with them as owner
3. User gets invited → Join existing org

**Tasks:**
- [ ] Create `POST /organizations` (create org)
- [ ] Create `POST /organizations/{org_id}/invite` (invite user)
- [ ] Create `POST /organizations/join/{invite_token}` (accept invite)
- [ ] Create `GET /organizations/me` (current org)
- [ ] Add email verification (optional)

---

#### Issue #14: Implement role-based access control

**Labels:** `P2-medium`, `feature`, `auth`

**Description:**
Enforce permissions based on user role and organization type.

**Organization Types:**
- `agency` - Team accounts with multiple members (Owner, Admin, Member roles)
- `individual` - Solo freelancer (single user, is their own admin)

**Roles:**

| Role | Org Type | Capabilities |
|------|----------|--------------|
| **Agency Owner** | `agency` | Full access + billing + user management + view team activities |
| **Agency Admin** | `agency` | Full access except billing + user management |
| **Agency Member** | `agency` | Own data only + profile setup + cannot access others' data |
| **Individual** | `individual` | Full access to own org + self-managed billing (no hierarchy above) |

**Permission checks:**
```python
@require_permission("training:write")
async def upload_training_data(...):
    ...

@require_permission("proposals:generate")
async def generate_proposal(...):
    ...

# Agency members can only access their own data
@require_permission("training:read", scope="own")
async def get_own_training_data(...):
    ...
```

**Permission Matrix:**

| Permission | Agency Owner | Agency Admin | Agency Member | Individual |
|------------|--------------|--------------|---------------|------------|
| `org:settings` | ✅ | ❌ | ❌ | ✅ |
| `org:billing` | ✅ | ❌ | ❌ | ✅ |
| `users:manage` | ✅ | ✅ | ❌ | N/A |
| `training:write` | ✅ all | ✅ all | ✅ own | ✅ own |
| `proposals:generate` | ✅ | ✅ | ✅ | ✅ |
| `analytics:view` | ✅ all | ✅ all | ✅ own | ✅ own |
| `team:activities` | ✅ | ✅ | ❌ | N/A |

**Tasks:**
- [ ] Create `require_permission` decorator with scope support
- [ ] Create permission check middleware
- [ ] Implement data scoping for Agency Members (access own data only)
- [ ] Update all routes with appropriate permissions
- [ ] Add tests for permission denial
- [ ] Add tests for data isolation between Agency Members

---

### Milestone: Sprint 5 - Testing & Documentation (Week 5)

---

#### Issue #15: Add integration tests for multi-tenant

**Labels:** `P1-high`, `test`

**Description:**
Verify tenant isolation works correctly.

**Test scenarios:**
- [ ] Org A cannot see Org B's training data
- [ ] Org A cannot query Org B's Pinecone vectors
- [ ] User can only access their org's data
- [ ] Owner can manage users, member cannot

---

#### Issue #16: Update API documentation

**Labels:** `P2-medium`, `docs`

**Description:**
Update OpenAPI/Swagger docs for new endpoints.

**Tasks:**
- [ ] Document all new organization endpoints
- [ ] Document all new user endpoints
- [ ] Document auth flow (JWT + API key)
- [ ] Add examples for multi-tenant usage
- [ ] Create Postman collection

---

#### Issue #17: Create data migration script

**Labels:** `P1-high`, `infra`

**Description:**
Script to migrate existing data to multi-tenant structure.

**Tasks:**
- [ ] Create default organization for existing data
- [ ] Migrate `user_profile` to `users`
- [ ] Add `org_id` to all existing documents
- [ ] Update Pinecone metadata with `org_id`
- [ ] Verify data integrity after migration
- [ ] Create rollback script

---

## Future Enhancements (Backlog)

These are NOT part of the initial 5 sprints but should be tracked.

---

#### Issue #18: Add Stripe billing integration

**Labels:** `P3-low`, `feature`

**Description:**
Integrate Stripe for subscription billing.

**Tasks:**
- [ ] Create Stripe customer on org creation
- [ ] Implement subscription checkout flow
- [ ] Handle webhooks (payment success/failure)
- [ ] Enforce plan limits

---

#### Issue #19: Add team analytics dashboard

**Labels:** `P3-low`, `feature`

**Description:**
Analytics per user within an organization.

**Metrics:**
- Proposals per user
- Hire rate per user
- Training data contributed per user

---

#### Issue #20: Add organization-level templates

**Labels:** `P3-low`, `feature`

**Description:**
Allow orgs to create custom proposal templates.

---

## Sprint Summary

| Sprint | Focus | Duration | Issues |
|--------|-------|----------|--------|
| **Sprint 1** | Code Cleanup | 1 week | #1, #2, #3, #4 |
| **Sprint 2** | Restructure | 1 week | #5, #6, #7 |
| **Sprint 3** | Multi-Tenant Foundation | 1 week | #8, #9, #10, #11 |
| **Sprint 4** | Auth & RBAC | 1 week | #12, #13, #14 |
| **Sprint 5** | Testing & Docs | 1 week | #15, #16, #17 |

**Total:** 5 weeks

---

## Quick Reference: First Actions

**Start with these today:**

1. Delete `proposal_generator.py` (Issue #1) - 5 minutes, safe
2. Create GitHub issues from this document
3. Assign Sprint 1 issues to this week

**DO NOT start:**
- New features
- More keyword lists
- More detection algorithms
- Frontend changes

Until the foundation is stable.
