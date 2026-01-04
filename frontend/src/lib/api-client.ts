/**
 * Simplified API Client
 * 
 * This replaces the complex caching logic in the old api.ts.
 * React Query handles ALL caching - this is just a thin fetch wrapper.
 * 
 * Features:
 * - Auth header injection
 * - Global error handling (401 redirects, 500 toasts)
 * - Request/response type safety
 * - NO custom caching (React Query does this)
 */

import { getToken, removeToken, isAuthenticated } from '@/services/auth';

const API_BASE = '/api';

/**
 * API Error class for structured error handling
 */
export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

/**
 * Error thrown when an authenticated API call is made without a valid token.
 */
export class AuthRequiredError extends Error {
  constructor(endpoint: string) {
    super(`Authentication required for ${endpoint}`);
    this.name = 'AuthRequiredError';
  }
}

interface RequestOptions extends Omit<RequestInit, 'body'> {
  /**
   * If true, check authentication before making the request.
   * Throws AuthRequiredError if not authenticated.
   */
  requireAuth?: boolean;
}

/**
 * Core fetch function with auth and error handling.
 * NO caching logic - React Query handles that.
 */
export async function apiGet<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { requireAuth = false, ...fetchOptions } = options;
  
  // Check auth before request to avoid 401 spam
  if (requireAuth && !isAuthenticated()) {
    throw new AuthRequiredError(endpoint);
  }
  
  const token = getToken();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...fetchOptions.headers,
  };
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'GET',
    ...fetchOptions,
    headers,
  });
  
  return handleResponse<T>(response);
}

/**
 * POST request
 */
export async function apiPost<T, D = unknown>(
  endpoint: string, 
  data?: D,
  options: RequestOptions = {}
): Promise<T> {
  const { requireAuth = false, ...fetchOptions } = options;
  
  if (requireAuth && !isAuthenticated()) {
    throw new AuthRequiredError(endpoint);
  }
  
  const token = getToken();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...fetchOptions.headers,
  };
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    ...fetchOptions,
    headers,
    body: data ? JSON.stringify(data) : undefined,
  });
  
  return handleResponse<T>(response);
}

/**
 * PUT request
 */
export async function apiPut<T, D = unknown>(
  endpoint: string, 
  data?: D,
  options: RequestOptions = {}
): Promise<T> {
  const { requireAuth = false, ...fetchOptions } = options;
  
  if (requireAuth && !isAuthenticated()) {
    throw new AuthRequiredError(endpoint);
  }
  
  const token = getToken();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...fetchOptions.headers,
  };
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'PUT',
    ...fetchOptions,
    headers,
    body: data ? JSON.stringify(data) : undefined,
  });
  
  return handleResponse<T>(response);
}

/**
 * PATCH request
 */
export async function apiPatch<T, D = unknown>(
  endpoint: string, 
  data?: D,
  options: RequestOptions = {}
): Promise<T> {
  const { requireAuth = false, ...fetchOptions } = options;
  
  if (requireAuth && !isAuthenticated()) {
    throw new AuthRequiredError(endpoint);
  }
  
  const token = getToken();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...fetchOptions.headers,
  };
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'PATCH',
    ...fetchOptions,
    headers,
    body: data ? JSON.stringify(data) : undefined,
  });
  
  return handleResponse<T>(response);
}

/**
 * DELETE request
 */
export async function apiDelete<T = void>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { requireAuth = false, ...fetchOptions } = options;
  
  if (requireAuth && !isAuthenticated()) {
    throw new AuthRequiredError(endpoint);
  }
  
  const token = getToken();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...fetchOptions.headers,
  };
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'DELETE',
    ...fetchOptions,
    headers,
  });
  
  return handleResponse<T>(response);
}

/**
 * Handle API response with global error handling
 */
async function handleResponse<T>(response: Response): Promise<T> {
  // Handle 401 - redirect to login
  if (response.status === 401) {
    removeToken();
    // Only redirect if not already on login page
    if (!window.location.pathname.includes('/login')) {
      window.location.href = '/login';
    }
    throw new ApiError('Authentication required', 401, 'AUTH_REQUIRED');
  }
  
  // Handle other errors
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Request failed' }));
    const message = errorData.message || errorData.detail || `HTTP ${response.status}`;
    throw new ApiError(message, response.status, errorData.code);
  }
  
  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }
  
  // Parse JSON response
  return response.json();
}

/**
 * Build URL with query parameters
 */
export function buildUrl(endpoint: string, params?: Record<string, string | number | boolean | undefined | null>): string {
  if (!params) return endpoint;
  
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      searchParams.append(key, String(value));
    }
  }
  
  const queryString = searchParams.toString();
  return queryString ? `${endpoint}?${queryString}` : endpoint;
}
