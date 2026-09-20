# Vercel & Supabase Deployment Guide (100% Free Tier)

Follow these exact click-by-click steps to deploy Cite RAG for free.

## 1. Supabase Setup
1. Go to [database.new](https://database.new) and sign in or create an account.
2. Create a new project:
   - **Name**: CiteRAG
   - **Database Password**: Generate a secure password and save it somewhere safe.
   - **Region**: Choose the one closest to you (or closest to your Vercel deployment region).
   - Click **Create new project**. Wait for the database to provision.
3. Open the **SQL Editor** from the left sidebar.
4. Click **New Query** and copy-paste the contents of `supabase/migrations/20231001000000_initial_schema.sql`. Click **Run**.
5. Create another query, paste the contents of `supabase/migrations/20231001000001_rate_limiting.sql`, and click **Run**.
6. Go to **Storage** in the left sidebar.
   - You should see a `documents` bucket. If not, create a new bucket named `documents` (keep "Public bucket" unchecked).
7. Go to **Project Settings** (the gear icon) > **API**.
   - Copy the `Project URL` (this is your `NEXT_PUBLIC_SUPABASE_URL`).
   - Copy the `anon` `public` key (this is your `NEXT_PUBLIC_SUPABASE_ANON_KEY`).
   - Copy the `service_role` `secret` key (this is your `SUPABASE_SERVICE_ROLE_KEY`). **NEVER COMMIT THIS**.

## 2. Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **Create API key**.
4. Copy the key (this is your `GEMINI_API_KEY`). **NEVER COMMIT THIS**.

## 3. GitHub Setup
1. Initialize a Git repository if you haven't already:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
2. Create a new repository on GitHub and push your code.
   - Ensure you do NOT commit your `.env.local` file! Check `.gitignore`.

## 4. Vercel Deployment
1. Go to [vercel.com/new](https://vercel.com/new) and sign in with GitHub.
2. Under "Import Git Repository", find your CiteRAG repository and click **Import**.
3. **Framework Preset**: Vercel should auto-detect **Next.js**.
4. **Root Directory**: Click Edit and select `frontend`.
5. **Environment Variables**: Add the following keys and the values you collected earlier:
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `GEMINI_API_KEY`
6. Click **Deploy**.

Vercel will build the Next.js app and assign you a free `*.vercel.app` domain. 

## Checklist of Secrets (DO NOT COMMIT)
- [ ] `SUPABASE_SERVICE_ROLE_KEY` (Gives admin access to your entire database)
- [ ] `GEMINI_API_KEY` (Can be used to incur LLM charges if leaked)
- [ ] Any `.env.local` or `.env` files. Ensure they are in `.gitignore`.
