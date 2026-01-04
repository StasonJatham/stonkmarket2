# Frontend Refactor Tracking

## Overview

**Goal:** Modernize the React SPA to use React 19 patterns, TanStack Query v5, and eliminate "Effect Spaghetti."

**Key Decisions:**
- ❌ **No WebSockets** - Prices update daily, not real-time
- ✅ **Single Cache Layer** - React Query ONLY (delete custom `apiCache`)
- ✅ **Zod Validation** - Runtime validation for all API responses
- ✅ **Feature-Based Architecture** - Domain-driven folder structure

---

## Phase 1: Foundation ✅ COMPLETE

### Step 1.1: Install Dependencies ✅
- [x] `@tanstack/react-query` v5
- [x] `@tanstack/react-query-devtools`
- [x] `zod`

### Step 1.2: Create API Client ✅
- [x] `lib/api-client.ts` - Axios/fetch with interceptors  
- [x] Global 401/500 error handling
- [ ] Remove ETag handling (React Query handles caching) - DEFERRED

### Step 1.3: Remove Custom Cache Layer
- [ ] Delete `frontend/src/lib/cache.ts` usage in `api.ts`
- [ ] Remove `etagStore`, `etagDataStore` from `api.ts`
- [ ] Remove `apiCache.fetch()` wrappers
> Note: Deferred - api.ts still has legacy code, but new queries bypass it

### Step 1.4: QueryClientProvider Setup ✅
- [x] `lib/query.tsx` - QueryProvider with DevTools
- [x] Configure `staleTime: 2min`, `gcTime: 10min` for daily data pattern
- [x] `refetchOnWindowFocus: true` to catch daily updates

### Step 1.5: Feature Folder Structure ✅
- [x] `features/market-data/` - queries.ts, schemas.ts
- [ ] `features/portfolio/` - needs queries.ts
- [x] `features/quant-engine/` - queries.ts
- [ ] `features/dip-swipe/` - needs queries.ts  
- [ ] `features/admin/` - needs queries.ts

---

## Phase 2: Data Fetching Migration - IN PROGRESS

### Priority Order (by complexity):
1. [x] `StockDetail.tsx` - Already uses `useStockDetail` from features ✅
2. [ ] `Landing.tsx` - Manual cache → useQuery
3. [x] `Dashboard.tsx` - Migrated to `useQuantRecommendations` ✅
4. [ ] `Portfolio.tsx` - 8+ useState → useQuery + useMutation
5. [ ] `DipSwipe.tsx` - Card fetching → useInfiniteQuery

### Context Deletion: ✅ COMPLETE
6. [x] DELETE `QuantContext.tsx` - Replaced with `useQuantRecommendations` ✅
7. [x] DELETE `DipContext.tsx` - Replaced with `useRanking` ✅

### Additional Migrations Completed:
8. [x] `Layout.tsx` - Migrated to `useRanking` for ticker component ✅

---

## Phase 3: Mutation Migration

- [ ] `ApiKeyManager.tsx` → useMutation
- [ ] `SuggestionManager.tsx` → useMutation
- [ ] `SchedulerPanel.tsx` → useMutation
- [ ] `Portfolio.tsx` CRUD → useMutation
- [ ] `SystemSettings.tsx` → useMutation
- [ ] `SuggestStockDialog.tsx` → useActionState (React 19)

---

## Phase 4: Memoization Cleanup

- [ ] Remove 18 unnecessary `useMemo` instances
- [ ] Remove 13 unnecessary `useCallback` instances
- [ ] Keep only: Recharts data, Framer Motion variants, external lib refs

---

## Phase 5: Component Atomization

- [ ] Split `StockDetailsPanel.tsx` (1500 lines → 5+ components)
- [ ] Split `Landing.tsx` (1600 lines → 8+ components)
- [ ] Split `Portfolio.tsx` (1900 lines → 10+ components)
- [ ] Create domain components (`PercentBadge`, `PriceDisplay`, etc.)

---

## Phase 6: TypeScript Hardening

- [ ] ESLint: `no-explicit-any` → `error`
- [ ] Add Zod schemas for ALL API types
- [ ] Enable `strict: true` if not already

---

## Cache Strategy (React Query)

```typescript
// Daily data pattern - prices update once per day
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,      // 5 minutes before refetch
      gcTime: 30 * 60 * 1000,        // 30 minutes in cache
      refetchOnWindowFocus: false,   // Don't spam API on tab switch
      retry: 1,                       // One retry on failure
    },
  },
});
```

---

## Progress Log

| Date | Phase | Task | Status |
|------|-------|------|--------|
| 2026-01-04 | 1 | Created tracking document | ✅ |
| 2026-01-04 | 1 | Install TanStack Query v5, devtools, zod | ✅ |
| 2026-01-04 | 1 | Create lib/api-client.ts with interceptors | ✅ |
| 2026-01-04 | 1 | Create lib/query.tsx QueryProvider | ✅ |
| 2026-01-04 | 1 | Create features/market-data/ structure | ✅ |
| 2026-01-04 | 1 | Create features/quant-engine/ structure | ✅ |
| 2026-01-04 | 2 | Migrate Dashboard.tsx to useQuantRecommendations | ✅ |
| 2026-01-04 | 2 | Migrate Layout.tsx to useRanking | ✅ |
| 2026-01-04 | 2 | DELETE QuantContext.tsx | ✅ |
| 2026-01-04 | 2 | DELETE DipContext.tsx | ✅ |
| 2026-01-04 | 2 | Fix Zod schema (add 'etf' type, updated_at field) | ✅ |
| 2026-01-04 | - | Fix Playwright tests (networkidle → element waiting) | ✅ |
| 2026-01-04 | - | **Playwright: 56/65 tests passing** | ✅ |

---

## Files to Delete

- [x] `frontend/src/context/QuantContext.tsx` ✅ DELETED
- [x] `frontend/src/context/DipContext.tsx` ✅ DELETED
- [ ] `frontend/src/lib/cache.ts` (after full migration)

## Files to Create

- [x] `frontend/src/lib/query.tsx` - QueryClientProvider with DevTools ✅
- [x] `frontend/src/lib/api-client.ts` - Axios client with interceptors ✅
- [x] `frontend/src/features/market-data/api/queries.ts` ✅
- [x] `frontend/src/features/market-data/api/schemas.ts` ✅
- [x] `frontend/src/features/quant-engine/api/queries.ts` ✅
- [x] `frontend/src/features/quant-engine/api/schemas.ts` ✅
- [ ] `frontend/src/app/providers.tsx`
- [ ] `frontend/src/features/portfolio/api/queries.ts`
- [ ] `frontend/src/features/dip-swipe/api/queries.ts`
- [ ] `frontend/src/features/admin/api/queries.ts`
- [ ] `frontend/src/features/*/api/mutations.ts`
