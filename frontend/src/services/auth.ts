/**
 * Authentication service with secure JWT handling
 */

const TOKEN_KEY = 'stonkmarket_token';
const API_BASE = '/api';

export interface AuthResponse {
  access_token: string;
  token_type: string;
  username: string;
  is_admin: boolean;
  mfa_required?: boolean;
}

export interface User {
  username: string;
  is_admin: boolean;
}

export interface MFAStatus {
  enabled: boolean;
  has_backup_codes: boolean;
  backup_codes_remaining: number | null;
}

export interface MFASetupResponse {
  provisioning_uri: string;
  secret: string;
  backup_codes: string[];
}

// Token management
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function removeToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export function isAuthenticated(): boolean {
  const token = getToken();
  if (!token) return false;
  
  // Check if token is expired
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.exp * 1000 > Date.now();
  } catch {
    return false;
  }
}

export function getAuthHeaders(): HeadersInit {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// Parse JWT token
export function parseToken(token: string): User | null {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return { 
      username: payload.sub,
      is_admin: payload.is_admin ?? false,
    };
  } catch {
    return null;
  }
}

export function getCurrentUser(): User | null {
  const token = getToken();
  if (!token) return null;
  if (!isAuthenticated()) {
    removeToken();
    return null;
  }
  return parseToken(token);
}

// API calls helper
async function authFetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Request failed' }));
    throw new Error(error.message || error.detail || `HTTP ${response.status}`);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

/**
 * Login with username and password
 * @throws Error with "MFA_REQUIRED" message if MFA code is needed
 */
export async function login(username: string, password: string, mfaCode?: string): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      username, 
      password,
      ...(mfaCode && { mfa_code: mfaCode }),
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Login failed' }));
    throw new Error(error.message || error.detail || 'Login failed');
  }

  const data: AuthResponse = await response.json();
  
  // Check if MFA is required (successful password, but need MFA code)
  if (data.mfa_required) {
    throw new Error('MFA_REQUIRED');
  }
  
  // Only set token if we got one (not MFA required)
  if (data.access_token) {
    setToken(data.access_token);
  }
  return data;
}

/**
 * Logout - revokes token on server
 */
export async function logout(): Promise<void> {
  try {
    await authFetch('/auth/logout', { method: 'POST' });
  } catch {
    // Ignore errors - still remove local token
  } finally {
    removeToken();
  }
}

/**
 * Logout from all devices
 */
export async function logoutAll(): Promise<void> {
  try {
    await authFetch('/auth/logout-all', { method: 'POST' });
  } catch {
    // Ignore errors - still remove local token
  } finally {
    removeToken();
  }
}

/**
 * Get current user info from server
 */
export async function getMe(): Promise<User> {
  return authFetch<User>('/auth/me');
}

/**
 * Change password and optionally username
 */
export async function changePassword(
  currentPassword: string,
  newPassword: string,
  newUsername?: string
): Promise<User> {
  return authFetch<User>('/auth/credentials', {
    method: 'PUT',
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
      new_username: newUsername,
    }),
  });
}

// MFA functions
export async function getMFAStatus(): Promise<MFAStatus> {
  return authFetch<MFAStatus>('/auth/mfa/status');
}

export async function setupMFA(): Promise<MFASetupResponse> {
  return authFetch<MFASetupResponse>('/auth/mfa/setup', { method: 'POST' });
}

export async function verifyMFA(code: string): Promise<{ enabled: boolean }> {
  return authFetch<{ enabled: boolean }>('/auth/mfa/verify', {
    method: 'POST',
    body: JSON.stringify({ code }),
  });
}

export async function disableMFA(code: string): Promise<void> {
  return authFetch('/auth/mfa/disable', {
    method: 'POST',
    body: JSON.stringify({ code }),
  });
}
