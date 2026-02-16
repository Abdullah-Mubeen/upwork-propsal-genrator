# GenProposal - Database Schema (v1.0 Draft)

> **Status:** Work in Progress  
> **Last Updated:** Feb 16, 2026  
> **Next Review:** TBD

---

## What We're Building

AI-powered proposal generator for Upwork freelancers & agencies. Users add their past projects, system finds similar work when they apply to new jobs, then writes personalized proposals in their voice.

---

## Database Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MongoDB (Primary DB)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   organizations ──┬── users                                             │
│        │          │                                                     │
│        └──────────┼── freelancer_profiles ──┬── portfolio_items         │
│                   │          │              │         │                 │
│                   │          └──────────────┼── job_preferences         │
│                   │                         │                           │
│                   │                         └── sent_proposals          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ portfolio_items.pinecone_id
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Pinecone (Vector DB)                            │
├─────────────────────────────────────────────────────────────────────────┤
│   Index: proposal-engine                                                │
│   Embedding: text-embedding-3-large (3072 dims)                         │
│   One vector per portfolio item                                         │
│   Metadata: org_id, profile_id, skills[], industry                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Collections & Fields

### 1. organizations
> Tenant accounts - individuals or agencies

| Field | Type | Description |
|-------|------|-------------|
| `org_id` | string | Primary key (org_abc123) |
| `name` | string | Company or person name |
| `org_type` | enum | `individual` or `agency` |
| `plan_tier` | enum | `free`, `starter`, `pro`, `enterprise` |
| `profile_limit` | int | Max profiles allowed (1, 3, 10, or -1 for unlimited) |
| `settings` | object | Custom preferences |
| `is_active` | bool | Soft delete flag |
| `created_at` | datetime | Registration timestamp |

---

### 2. users
> Login credentials & role-based access

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Primary key (usr_abc123) |
| `org_id` | string | Foreign key → organizations |
| `email` | string | Login identifier |
| `name` | string | Display name |
| `role` | enum | `super_admin`, `admin`, `member` |
| `api_key_hash` | string | SHA256 hash of API key |
| `api_key_prefix` | string | First 8 chars for identification |
| `is_active` | bool | Account status |
| `last_login` | datetime | Last activity |

---

### 3. freelancer_profiles
> Bio, skills, voice for proposal personalization

| Field | Type | Description |
|-------|------|-------------|
| `profile_id` | string | Primary key (prof_abc123) |
| `org_id` | string | Foreign key → organizations |
| `name` | string | Freelancer name |
| `title` | string | Professional headline |
| `bio` | string | Professional summary (~500 chars) |
| `skills` | array | List of skills |
| `hourly_rate` | float | Rate in USD |
| `years_experience` | int | Years in field |
| `source` | enum | `manual` or `upwork` |
| `source_urls` | array | Import URLs if any |
| `is_active` | bool | Can be used for proposals |

---

### 4. portfolio_items
> Past projects - the "proof" in proposals

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | string | Primary key (port_abc123) |
| `org_id` | string | Foreign key → organizations |
| `profile_id` | string | Foreign key → freelancer_profiles |
| `project_title` | string | Project name |
| `deliverables` | string | What was built |
| `skills` | array | Technologies used |
| `outcome` | string | Results achieved |
| `portfolio_url` | string | Live project link |
| `industry` | string | SaaS, FinTech, HealthTech, etc. |
| `client_feedback` | string | Testimonial quote |
| `duration_days` | int | Project length |
| `pinecone_id` | string | Vector ID in Pinecone |
| `is_embedded` | bool | Has vector? |

---

### 5. job_preferences
> Filters for which jobs to show/apply to

| Field | Type | Description |
|-------|------|-------------|
| `pref_id` | string | Primary key (pref_abc123) |
| `org_id` | string | Foreign key → organizations |
| `profile_id` | string | Foreign key → freelancer_profiles |
| `payment_verified` | bool | Only verified clients? |
| `min_budget` | float | Minimum job budget |
| `max_budget` | float | Maximum job budget |
| `experience_levels` | array | `entry`, `intermediate`, `expert` |
| `job_types` | array | `fixed`, `hourly` |
| `categories` | array | Job categories |
| `excluded_keywords` | array | Words to skip |
| `min_client_rating` | float | Minimum client stars |

---

### 6. sent_proposals
> Track what was sent & outcomes

| Field | Type | Description |
|-------|------|-------------|
| `proposal_id` | string | Primary key (prop_abc123) |
| `org_id` | string | Foreign key → organizations |
| `profile_id` | string | Foreign key → freelancer_profiles |
| `job_title` | string | Job applied to |
| `proposal_text` | string | Generated content |
| `word_count` | int | Proposal length |
| `outcome` | enum | `sent`, `viewed`, `hired`, `rejected` |
| `portfolio_used` | array | Portfolio links included |
| `confidence_score` | float | Quality score (0-1) |
| `sent_at` | datetime | When submitted |

---

## Plan Limits

| Plan | Profile Limit | For |
|------|--------------|-----|
| Free | 1 | Individual freelancers |
| Starter | 3 | Small agencies |
| Pro | 10 | Growing agencies |
| Enterprise | Unlimited | Large agencies |

---

## Proposal Generation Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  1. JOB INPUT         2. EMBED              3. VECTOR SEARCH             │
│  ┌─────────┐         ┌─────────┐           ┌─────────────────┐           │
│  │ Job     │ ──────► │ OpenAI  │ ────────► │ Pinecone        │           │
│  │ Posting │         │ Embed   │           │ (filter: org_id)│           │
│  └─────────┘         └─────────┘           └────────┬────────┘           │
│                                                     │                    │
│                                            Top 5 similar projects        │
│                                                     │                    │
│                                                     ▼                    │
│  5. GENERATE          4. ENRICH            ┌─────────────────┐           │
│  ┌─────────┐         ┌─────────┐           │ MongoDB         │           │
│  │ GPT-4o  │ ◄────── │ Build   │ ◄──────── │ Full portfolio  │           │
│  │ Output  │         │ Prompt  │           │ + Profile bio   │           │
│  └─────────┘         └─────────┘           └─────────────────┘           │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────┐                            │
│  │ Personalized Proposal                    │                            │
│  │ - Hook (addresses their problem)         │                            │
│  │ - Proof (similar past projects)          │                            │
│  │ - Approach (how you'll solve it)         │                            │
│  │ - CTA (friendly close)                   │                            │
│  └─────────────────────────────────────────┘                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Key Relationships

- **Organization → Profile:** Individual = 1 profile, Agency = multiple (based on plan)
- **Profile → Portfolio:** Each profile has their own past projects
- **Profile → Job Preferences:** Each profile can have different job filters
- **Portfolio Item → Pinecone Vector:** 1:1 mapping for semantic search

---

## Coming Soon

- [ ] Portfolio bulk import from CSV
- [ ] Upwork profile auto-sync
- [ ] Job ingestion module
- [ ] Analytics dashboard
- [ ] Webhook integrations

---

## Questions?

Reach out to the dev team. This doc will be updated as we ship new features.
