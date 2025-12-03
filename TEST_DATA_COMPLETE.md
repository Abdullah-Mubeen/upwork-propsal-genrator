# 🧪 Complete Test Data Suite

## Test Scenario 1: Urgent Backend API (High Priority, Fast Turnaround)

```json
{
  "company_name": "FinTech Innovations Inc",
  "job_title": "Senior Backend Engineer - Urgent",
  "job_description": "URGENT: We have a critical payment processing API that needs optimization and bug fixes. The system handles 10k+ transactions daily but recently started experiencing timeouts during peak hours. We need someone who can diagnose the bottleneck, implement caching, and optimize database queries. This is a 3-day emergency project that will directly impact our revenue.",
  "your_proposal_text": "I understand the urgency and have extensive experience with payment systems. I've optimized 5+ financial platforms handling millions in daily transactions. My approach: 1) Run APM tools (New Relic/DataDog) to identify bottlenecks, 2) Implement Redis caching for frequently accessed data, 3) Optimize slow queries with proper indexing, 4) Add rate limiting to prevent cascading failures. I can start immediately and provide daily updates. I've solved similar issues in 2-3 days before. Expect 30-40% improvement in response time.",
  "skills_required": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes", "Performance Optimization", "Payment Systems"],
  "industry": "Finance",
  "task_type": "backend_api",
  "project_status": "completed",
  "urgent_adhoc": true,
  "has_feedback": true,
  "start_date": "2024-12-02",
  "end_date": "2024-12-05",
  "portfolio_url": "https://github.com/yourname/fintech-optimization",
  "client_feedback": "Outstanding work! They diagnosed the issue in 2 hours and had a fix deployed by day 1. The system now handles 3x the load. Highly responsive, great communicator. Would hire again immediately."
}
```

**Expected: 5 chunks, 5 embeddings, 5 Pinecone vectors**

---

## Test Scenario 2: Full Stack Website Build (40-Day Project)

```json
{
  "company_name": "BrandCo Marketing Agency",
  "job_title": "Full Stack Developer - Company Website",
  "job_description": "We need a complete modern website for our marketing agency. The site should showcase our services, team, portfolio, blog, and contact forms. Requirements: responsive design (mobile, tablet, desktop), SEO optimized, fast loading times, CMS integration for blog posts, newsletter signup, client testimonials section, image gallery, service comparison tool, and appointment booking calendar integration. Must be built on modern tech stack with proper documentation. Timeline: 40 days.",
  "your_proposal_text": "I specialize in full-stack web development and have built 20+ agency websites. I'll use Next.js for the frontend (React, Tailwind CSS, TypeScript), Node.js with Express for the backend, PostgreSQL for database, and deploy on Vercel + AWS. My approach: 1) Design UI/UX mockups in Figma (3 days), 2) Frontend development with responsive components (12 days), 3) Backend API with authentication (8 days), 4) CMS integration with Sanity (5 days), 5) Testing and optimization (8 days), 6) Deployment and documentation (4 days). The site will have <2s load time, 95+ Lighthouse score, and full SEO optimization.",
  "skills_required": ["React", "Next.js", "TypeScript", "Node.js", "Express", "PostgreSQL", "Tailwind CSS", "REST API", "SEO", "DevOps"],
  "industry": "Marketing",
  "task_type": "full_stack",
  "project_status": "completed",
  "urgent_adhoc": false,
  "has_feedback": true,
  "start_date": "2024-11-01",
  "end_date": "2024-12-10",
  "portfolio_url": "https://github.com/yourname/brandco-website",
  "client_feedback": "Fantastic team! They delivered exactly what we needed. The website looks professional, loads incredibly fast, and our SEO rankings improved significantly within 2 weeks. Great attention to detail. Exceeded expectations."
}
```

**Expected: 5 chunks, 5 embeddings, 5 Pinecone vectors**

---

## Test Scenario 3: Mobile App Development (90-Day Project)

```json
{
  "company_name": "FitTrack Inc",
  "job_title": "Mobile App Developer - React Native",
  "job_description": "We're building a fitness tracking and coaching app for iOS and Android. Features required: user authentication, workout logging, progress tracking, photo gallery, personal coach messaging, workout video streaming, social sharing, in-app notifications, payment processing for premium features, and analytics dashboard. The app needs offline support, smooth animations, and excellent performance. 90-day timeline with 2 releases (MVP in 60 days, full release in 90 days).",
  "your_proposal_text": "I've developed 8 React Native apps with combined 500k+ downloads. I recommend: 1) React Native with Expo for cross-platform code, 2) Firebase for real-time backend and messaging, 3) Redux for state management, 4) AWS S3 for video storage and streaming. Sprint breakdown: Weeks 1-2 (Auth & user profiles), Weeks 3-4 (Workout logging & data), Weeks 5-6 (Coach messaging & notifications), Weeks 7-8 (Payment integration), Weeks 9-10 (Video streaming optimization), Weeks 11-12 (Analytics & polish). I'll deliver a high-quality app with <100MB bundle size and 60 FPS animations.",
  "skills_required": ["React Native", "JavaScript", "Firebase", "Redux", "AWS", "Payment APIs", "Video Streaming", "Mobile UI/UX"],
  "industry": "Health & Fitness",
  "task_type": "mobile_app",
  "project_status": "ongoing",
  "urgent_adhoc": false,
  "has_feedback": false,
  "start_date": "2024-10-15",
  "end_date": "2025-01-13",
  "portfolio_url": "https://apps.apple.com/us/developer/yourname/id1234567890",
  "client_feedback": null
}
```

**Expected: 5 chunks, 5 embeddings, 5 Pinecone vectors (no feedback)**

---

## Test Scenario 4: Emergency Frontend Fix (2-Day Urgent)

```json
{
  "company_name": "StartupXYZ",
  "job_title": "Frontend Emergency - Bug Fixes & Performance",
  "job_description": "Critical production issue: Our React dashboard is crashing during peak usage, showing 'Maximum call stack size exceeded' errors. Users are unable to access reports, which is blocking business operations. We need immediate diagnosis and fixes. The codebase uses React 18, Redux, and TypeScript. Also need to investigate why page load time increased from 1.2s to 4.5s over the last week. Need this resolved ASAP, ideally within 2 days.",
  "your_proposal_text": "I specialize in React performance optimization and debugging. I'll start with: 1) Profile React render cycles using React DevTools and Chrome DevTools to find infinite loops, 2) Check for memory leaks in Redux store subscriptions, 3) Analyze bundle size for unexpected increases, 4) Review recent commits for the bottleneck introduction. Common causes: unnecessary re-renders, missing keys in lists, memory leaks. I've fixed similar issues in 4-8 hours. I'll provide detailed analysis report, root cause, and permanent fix with tests to prevent recurrence.",
  "skills_required": ["React", "TypeScript", "Redux", "Performance Debugging", "Chrome DevTools", "Profiling"],
  "industry": "Technology",
  "task_type": "frontend",
  "project_status": "completed",
  "urgent_adhoc": true,
  "has_feedback": true,
  "start_date": "2024-12-03",
  "end_date": "2024-12-05",
  "portfolio_url": "https://github.com/yourname/react-performance",
  "client_feedback": "Lifesaver! They found the problem within 2 hours - we had accidentally added a memory leak in Redux middleware. Fixed it, deployed, and everything works perfectly now. Response time back to 1.1s. What would have taken us days was done in hours. Highly professional."
}
```

**Expected: 5 chunks, 5 embeddings, 5 Pinecone vectors**

---

## Test Scenario 5: Consultation/Advisory (Non-Technical)

```json
{
  "company_name": "EnterpriseCorp Global",
  "job_title": "Tech Architecture Consultation",
  "job_description": "We're planning a complete digital transformation and need expert guidance on technology architecture. Our legacy monolith needs to be modernized. Looking for someone to: 1) Audit current system, 2) Recommend microservices architecture, 3) Plan migration strategy, 4) Design cloud infrastructure (AWS/Azure), 5) Create implementation roadmap. This is advisory/consulting work, 10-15 hours over 2 weeks.",
  "your_proposal_text": "I've guided 15+ enterprises through digital transformations. My approach: 1) Week 1: System audit, dependency analysis, identify migration candidates, 2) Week 2: Design microservices boundaries, create architecture diagrams, build AWS deployment strategy. I'll deliver: architecture decision records (ADRs), detailed roadmap, risk analysis, team training recommendations, and cost projections. Expected outcome: clear modernization path that reduces tech debt by 60% and improves deployment frequency from quarterly to daily.",
  "skills_required": ["System Architecture", "Microservices", "AWS", "Technical Leadership", "Strategy"],
  "industry": "Enterprise",
  "task_type": "consultation",
  "project_status": "completed",
  "urgent_adhoc": false,
  "has_feedback": true,
  "start_date": "2024-11-20",
  "end_date": "2024-12-03",
  "portfolio_url": "https://linkedin.com/in/yourname",
  "client_feedback": "Exceptional insights. Their recommendations guided our entire transformation. The roadmap was realistic and detailed. Their team was knowledgeable, collaborative, and helped us avoid costly mistakes. Exactly what we needed."
}
```

**Expected: 5 chunks, 5 embeddings, 5 Pinecone vectors**

---

## Testing Checklist

### Step 1: Start the Server
```bash
cd /home/abdullah-mubeen/Project/Upwork-Proposal_Generator
source env/bin/activate
python main.py
```

Server will run on `http://localhost:8000`

### Step 2: Test Each Scenario

For each test data above:

1. **Go to Swagger UI**: `http://localhost:8000/docs`
2. **Find endpoint**: `POST /api/v1/jobs/upload`
3. **Click "Try it out"**
4. **Paste entire JSON** from test scenario
5. **Click "Execute"**

### Step 3: Verify Response

Should see:
```
✅ Status: 201 Created
✅ Message: "Job data uploaded successfully! Processed: 5 chunks, 5 embeddings, saved to 5 Pinecone vectors"
✅ contract_id: job_<8-chars> (e.g., job_a1b2c3d4)
✅ Response includes all 13 fields
```

### Step 4: Check MongoDB

```bash
# Connect to MongoDB
mongo mongodb://localhost:27017/proposal_generator

# Check training_data
db.training_data.find().pretty()

# Expected fields
{
  _id: ObjectId(...),
  contract_id: "job_a1b2c3d4",
  company_name: "...",
  job_title: "...",
  has_feedback: true/false,
  urgent_adhoc: true/false,
  created_at: ISODate(...),
  updated_at: ISODate(...)
}

# Check chunks (should be 5)
db.chunks.find({contract_id: "job_a1b2c3d4"}).count()

# Check embeddings (should be 5)
db.embeddings.find({contract_id: "job_a1b2c3d4"}).count()

# Check feedback_data (only if has_feedback=true)
db.feedback_data.find({contract_id: "job_a1b2c3d4"}).pretty()
```

### Step 5: Check Pinecone

```bash
# Use Pinecone console or API
# Check vectors stored with contract_id metadata
```

### Step 6: Verify Pipeline Logs

Look for logs like:
```
🚀 STARTING COMPLETE TRAINING PIPELINE
✓ Job stored with contract_id: job_a1b2c3d4
✓ Created 5 smart chunks
✓ Generated 5 embeddings
✓ Feedback saved to collection
✓ Saved 5 vectors to Pinecone
✅ PIPELINE COMPLETE
```

---

## Success Criteria

After running all 5 test scenarios, you should have:

- ✅ **5 jobs** in MongoDB `training_data` collection
- ✅ **25 chunks** total in `chunks` collection (5 per job)
- ✅ **25 embeddings** in `embeddings` collection
- ✅ **25 Pinecone vectors** with rich metadata
- ✅ **3 feedback records** in `feedback_data` collection (scenarios 1, 2, 4, 5 have feedback)
- ✅ **All contract_ids** in format `job_<8-chars>`
- ✅ **has_feedback flag** correctly set (true for 4 jobs, false for 1)
- ✅ **No errors** in logs

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| **Contract_id is full UUID** | Update `app/db.py` line 183 (should use `uuid.uuid4().hex[:8]`) |
| **has_feedback always false** | Check `job_data_schema.py` validator is applied |
| **Feedback not in MongoDB** | Check logs, verify `client_feedback` field is not empty |
| **Pinecone vectors not saving** | Check `PINECONE_API_KEY` and `PINECONE_HOST` in `.env` |
| **Chunks not created** | Check `job_description` and `your_proposal_text` are long enough |
| **Embeddings generation fails** | Check `OPENAI_API_KEY` is valid in `.env` |
| **Server won't start** | Check all environment variables in `.env` are set |

---

## Expected Performance

| Operation | Time |
|-----------|------|
| Store job data | <100ms |
| Create 5 chunks | <50ms |
| Generate 5 embeddings | 2-5 seconds (OpenAI API latency) |
| Save to Pinecone | <1 second |
| **Total pipeline** | **3-7 seconds** |

---

## Data Retention

All data persists in:
- **MongoDB**: Indefinitely (or until manually deleted)
- **Pinecone**: Indefinitely (or until manually deleted)

To clean up test data:
```bash
# MongoDB cleanup
db.training_data.deleteMany({contract_id: /^job_/})
db.chunks.deleteMany({contract_id: /^job_/})
db.embeddings.deleteMany({contract_id: /^job_/})
db.feedback_data.deleteMany({contract_id: /^job_/})
```

---

**Test Suite Version**: 2.0 (Complete Pipeline)  
**Created**: December 4, 2024  
**Status**: ✅ Ready for Testing
