# Multi-Tenant Architecture Redesign

**Version:** 2.0  
**Status:** Proposed  
**Last Updated:** February 2026

---

## Goals

1. **Clean separation of concerns** - Routes thin, services focused
2. **Hybrid multi-tenant support** - Agency (multiple users) + Individual (isolated)
3. **Remove technical debt** - Delete unused code, consolidate duplicates
4. **Prepare for scale** - Proper data isolation, RBAC foundation
5. **Maintainability** - New features don't break existing ones

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TARGET ARCHITECTURE v2.0                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  API LAYER (Thin Controllers)                                               │
│  ├── api/v1/                                                                │
│  │   ├── proposals.py      (50-100 lines - just routing)                   │
│  │   ├── training.py       (50-100 lines)                                  │
│  │   ├── analytics.py      (50-100 lines)                                  │
│  │   ├── admin.py          (50-100 lines)                                  │
│  │   └── organizations.py  (NEW - org management)                          │
│  │                                                                          │
│  ├── middleware/                                                            │
│  │   ├── auth.py           (JWT + API key validation)                      │
│  │   └── tenant.py      (NEW - inject tenant context)                      │
│                                                                             │
│  SERVICE LAYER (Business Logic)                                             │
│  ├── services/                                                              │
│  │   ├── proposal_service.py     (~200 lines - orchestrates generation)   │
│  │   ├── training_service.py     (~200 lines - data ingestion)            │
│  │   ├── retrieval_service.py    (~300 lines - vector search)             │
│  │   ├── embedding_service.py    (~150 lines - OpenAI embeddings)         │
│  │   ├── analytics_service.py    (~150 lines - stats & reports)           │
│  │   └── organization_service.py (NEW - org CRUD)                          │
│                                                                             │
│  DOMAIN LAYER (Pure Models)                                                 │
│  ├── domain/                                                                │
│  │   ├── organization.py   (NEW - Pydantic models)                         │
│  │   ├── user.py           (NEW - user models)                             │
│  │   ├── training_data.py  (cleaned schemas)                               │
│  │   ├── proposal.py       (proposal models)                               │
│  │   └── constants.py   (ALL keywords in one place)                        │
│                                                                             │
│  INFRASTRUCTURE LAYER (External Systems)                                    │
│  ├── infra/                                                                 │
│  │   ├── mongodb/                                                           │
│  │   │   ├── connection.py         (connection management)                 │
│  │   │   ├── base_repository.py    (generic CRUD)                          │
│  │   │   └── repositories/                                                  │
│  │   │       ├── training_repo.py                                          │
│  │   │       ├── proposal_repo.py                                          │
│  │   │       ├── analytics_repo.py                                         │
│  │   │       └── organization_repo.py (NEW)                                │
│  │   │                                                                      │
│  │   ├── pinecone/                                                          │
│  │   │   └── vector_store.py       (tenant-aware queries)                  │
│  │   │                                                                      │
│  │   └── openai/                                                            │
│  │       ├── llm_client.py         (text generation)                       │
│  │       └── embedding_client.py   (embeddings)                            │
│                                                                             │
│  UTILS (Truly Generic)                                                      │
│  ├── utils/                                                                 │
│  │   ├── chunker.py         (simplified - one implementation)              │
│  │   ├── prompt_builder.py  (just prompt templates)                        │
│  │   └── text_analysis.py   (pain points, urgency - ONE place)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Tenant Data Model

### New Collections

#### 1. `organizations` (NEW)

```javascript
{
  "_id": ObjectId,
  "org_id": "org_abc123",           // Unique, URL-safe ID
  "name": "Acme Agency",
  "slug": "acme-agency",            // For subdomains/URLs
  "type": "agency",                 // "agency" | "individual"
  "owner_id": "user_xyz",           // FK to users
  
  "settings": {
    "default_style": "professional",
    "default_tone": "confident",
    "max_word_count": 350
  },
  
  "limits": {
    "max_users": 10,                // -1 for unlimited
    "max_training_data": 1000,
    "max_proposals_per_day": 100,
    "max_pinecone_vectors": 50000
  },
  
  "pinecone_config": {
    "namespace": "org_abc123",      // Dedicated namespace
    "vectors_count": 0
  },
  
  "billing": {
    "plan": "agency_pro",           // free | pro | agency | enterprise
    "stripe_customer_id": "cus_xxx",
    "subscription_id": "sub_xxx",
    "current_period_end": ISODate
  },
  
  "created_at": ISODate,
  "updated_at": ISODate,
  "is_active": true
}
```

#### 2. `users` (REPLACES `user_profile`)

```javascript
{
  "_id": ObjectId,
  "user_id": "user_xyz789",
  "org_id": "org_abc123",           // FK to organizations
  "email": "john@acme.com",
  "password_hash": "...",           // If using password auth
  "name": "John Doe",
  
  // Role depends on org.type:
  // - For agency orgs: "owner" | "admin" | "member"
  // - For individual orgs: always "owner" (they are their own admin)
  "role": "admin",
  
  "permissions": [                  // Granular permissions (derived from role)
    "training:write",
    "proposals:generate", 
    "analytics:view"
  ],
  
  "profile": {
    "upwork_url": "https://upwork.com/...",
    "bio": "Full-stack developer...",
    "hourly_rate": 75,
    "skills": ["React", "Node.js"],
    "timezone": "America/New_York",
    "proposal_template_preferences": {
      "default_style": "conversational",
      "include_portfolio": true,
      "signature": "Let's chat!"
    }
  },
  
  "api_key_hash": "...",            // Per-user API key (optional)
  
  "created_at": ISODate,
  "last_login": ISODate,
  "is_active": true
}

// NOTE: For Individual (Self-Managed) accounts:
// - org.type = "individual"
// - user.role = "owner" (always)
// - User manages their own billing (no external admin)
// - org.owner_id = user.user_id (same person)
```

#### 3. Updated Existing Collections

ALL existing collections get `org_id`:

```javascript
// training_data - UPDATED
{
  "org_id": "org_abc123",           // NEW - required
  "contract_id": "job_xxx",
  "company_name": "...",
  // ... rest unchanged
}

// chunks - UPDATED
{
  "org_id": "org_abc123",           // NEW - required
  "chunk_id": "...",
  // ... rest unchanged
}

// sent_proposals - UPDATED
{
  "org_id": "org_abc123",           // NEW - required
  "user_id": "user_xyz",            // NEW - who sent it
  "proposal_id": "...",
  // ... rest unchanged
}
```

---

## Tenant Isolation Strategy

### MongoDB Isolation

```python
# All queries automatically include org_id filter

class TenantAwareRepository:
    def __init__(self, org_id: str):
        self.org_id = org_id
    
    def find(self, query: dict) -> List[dict]:
        # Automatically inject org_id
        query["org_id"] = self.org_id
        return self.collection.find(query)
    
    def insert(self, document: dict) -> str:
        # Automatically add org_id
        document["org_id"] = self.org_id
        return self.collection.insert_one(document)
```

### Pinecone Isolation

**Strategy: Hybrid Namespace + Metadata**

```python
# For paying orgs: dedicated namespace
namespace = f"org_{org_id}"

# For free tier: shared namespace with metadata filter
filter = {"org_id": {"$eq": org_id}}
```

---

## RBAC Design (Future-Ready)

### Organization Types

| Type | Description | Users | Billing |
|------|-------------|-------|---------|
| **`agency`** | Team account with multiple members | Owner + Admin(s) + Member(s) | Managed by Owner/Admin |
| **`individual`** | Solo freelancer account | Single user (is their own admin) | Self-managed |

### Roles & Permissions

#### 1. Agency Owner / Admin (for `agency` type orgs)

Full access to the organization:
- Manage billing and subscriptions
- Manage users and assign roles  
- View all team activities and analytics
- Configure member profiles, permissions, and settings
- Full CRUD on all training data, proposals, sent history

#### 2. Agency Member (for `agency` type orgs)

Limited access within the agency:
- Access only their **own data**
- Manage their own training data, proposals, and analytics
- Setup personal profile (skills, Upwork URL, imports, proposal template structure)
- **Cannot** access other members' data unless explicitly shared
- **Cannot** manage billing or invite users

#### 3. Individual / Self-Managed (for `individual` type orgs)

Solo freelancer - they ARE the admin:
- Full access to their own organization
- Manage their own billing and subscription
- All training data, proposals, analytics are theirs alone
- **Not associated** with any agency
- Equivalent to Agency Owner but for a single-user org

### Role Hierarchy

```
Platform Super Admin (you - system owner)
├── Can: Everything + platform config + all orgs

Agency Owner (org.type = "agency")
├── Can: Full org access + billing + subscriptions
├── Can: Manage users, assign roles
├── Can: View all team activities
├── Can: Configure member profiles/permissions

Agency Admin (org.type = "agency")
├── Can: Same as Owner EXCEPT billing management
├── Can: Manage users, invite members
├── Can: Full CRUD on training data + proposals

Agency Member (org.type = "agency")
├── Can: Own training data + proposals + analytics
├── Can: Setup own profile
├── Cannot: Access other members' data
├── Cannot: Manage users or billing

Individual (org.type = "individual")
├── Can: Everything for their single-user org
├── Can: Manage own billing (self-service)
├── Is: Their own admin (no hierarchy above)
```

### Permission Matrix

| Permission | Agency Owner | Agency Admin | Agency Member | Individual |
|------------|--------------|--------------|---------------|------------|
| `org:settings` | ✅ | ❌ | ❌ | ✅ |
| `org:billing` | ✅ | ❌ | ❌ | ✅ |
| `users:manage` | ✅ | ✅ | ❌ | N/A |
| `users:invite` | ✅ | ✅ | ❌ | N/A |
| `training:write` | ✅ all | ✅ all | ✅ own | ✅ own |
| `training:read` | ✅ all | ✅ all | ✅ own | ✅ own |
| `proposals:generate` | ✅ | ✅ | ✅ | ✅ |
| `proposals:history` | ✅ all | ✅ all | ✅ own | ✅ own |
| `analytics:view` | ✅ all | ✅ all | ✅ own | ✅ own |
| `analytics:export` | ✅ | ✅ | ❌ | ✅ |
| `team:activities` | ✅ | ✅ | ❌ | N/A |

---

## Migration Strategy

### Phase 1: Cleanup (No Breaking Changes)

1. Delete `proposal_generator.py`
2. Consolidate constants into `domain/constants.py`
3. Consolidate text analysis into `utils/text_analysis.py`
4. Remove `data_chunker.py` wrapper

### Phase 2: Restructure (Internal Changes)

1. Split `db.py` into repositories
2. Create service layer
3. Thin out route files
4. Add comprehensive tests

### Phase 3: Multi-Tenant Foundation

1. Create `organizations` collection
2. Create `users` collection migrating from `user_profile`
3. Add `org_id` to all collections
4. Create migration script for existing data
5. Update Pinecone namespace strategy

### Phase 4: RBAC & Auth

1. Implement JWT authentication
2. Create organization registration flow
3. Implement role-based middleware
4. Add audit logging

---

## File Structure After Cleanup

```
app/
├── __init__.py
├── main.py                      # FastAPI app initialization
├── config.py                    # Settings (unchanged)
│
├── api/                         # NEW - API layer
│   ├── __init__.py
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── proposals.py         # Thin route handlers
│   │   ├── training.py
│   │   ├── analytics.py
│   │   ├── admin.py
│   │   ├── organizations.py     # NEW
│   │   └── users.py             # NEW
│   │
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py              # JWT + API key auth
│       └── tenant.py            # NEW - tenant context injection
│
├── services/                    # NEW - Business logic
│   ├── __init__.py
│   ├── proposal_service.py
│   ├── training_service.py
│   ├── retrieval_service.py
│   ├── embedding_service.py
│   ├── analytics_service.py
│   └── organization_service.py  # NEW
│
├── domain/                      # NEW - Pure models
│   ├── __init__.py
│   ├── constants.py             # ALL keywords consolidated
│   ├── organization.py          # NEW
│   ├── user.py                  # NEW
│   ├── training_data.py
│   ├── proposal.py
│   └── enums.py
│
├── infra/                       # NEW - External systems
│   ├── __init__.py
│   ├── mongodb/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── base_repository.py
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── training_repo.py
│   │       ├── proposal_repo.py
│   │       ├── sent_proposal_repo.py
│   │       ├── analytics_repo.py
│   │       └── organization_repo.py  # NEW
│   │
│   ├── pinecone/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   │
│   └── openai/
│       ├── __init__.py
│       ├── llm_client.py
│       └── embedding_client.py
│
└── utils/                       # Simplified utilities
    ├── __init__.py
    ├── chunker.py               # Single chunking implementation
    ├── prompt_builder.py        # Simplified prompt templates
    └── text_analysis.py         # Pain points, urgency, industry detection
```

### Files to DELETE

```
app/utils/
├── proposal_generator.py        # ✅ DELETED
├── data_chunker.py              # ✅ DELETED
├── advanced_chunker.py          # ✅ DELETED (stub in job_data_processor.py)
├── hook_strategy.py             # MERGE into prompt_builder.py
└── metadata_extractor.py        # MERGE into text_analysis.py

# After restructure, these become obsolete:
app/routes/                      # MOVE to api/v1/
app/models/                      # MOVE to domain/
```

---

## Estimated Effort

| Phase | Tasks | Effort | Risk |
|-------|-------|--------|------|
| **Phase 1** | Cleanup | 2-3 days | Low |
| **Phase 2** | Restructure | 5-7 days | Medium |
| **Phase 3** | Multi-Tenant | 5-7 days | Medium |
| **Phase 4** | RBAC & Auth | 3-5 days | Medium |

**Total:** ~3-4 weeks of focused work

---

## Success Metrics

After implementation:

- [ ] Total lines of code reduced by 30%+
- [ ] Zero code duplication for constants/analysis
- [ ] All collections have `org_id` index
- [ ] Pinecone queries isolated by tenant
- [ ] New features can be added without refactoring
- [ ] Test coverage > 70%

---

## Next Steps

1. Review and approve this architecture
2. Create GitHub issues from `GITHUB_ISSUES.md`
3. Start with Phase 1 (no breaking changes)
4. Iterate through phases
