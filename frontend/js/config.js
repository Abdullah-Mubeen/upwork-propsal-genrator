/**
 * API Configuration
 * Change the BASE_URL here to switch between development and production
 * 
 * ⚠️  IMPORTANT: Before final deployment, ensure production URL is active!
 */

// ===== CHANGE THIS LINE TO SWITCH ENVIRONMENTS =====
// const BASE_URL = 'https://ml.genproposals.com';  // PRODUCTION
const BASE_URL = 'http://localhost:8000';           // LOCAL DEV - switch before deploy!
// ===================================================

// Main API endpoint
const API = `${BASE_URL}/api`;

// Specific API endpoints for each module
const AUTH_API = `${BASE_URL}/api/auth/verify`;
const JOB_DATA_API = `${BASE_URL}/api/job-data`;
const PROPOSALS_API = `${BASE_URL}/api/proposals`;
const ANALYTICS_API = `${BASE_URL}/api/analytics`;
const ADMIN_API = `${BASE_URL}/api/admin`;
const PROFILE_API = `${BASE_URL}/api/profile`;
