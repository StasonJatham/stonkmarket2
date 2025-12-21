const TOKEN_KEY = 'stonkmarket_token';

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  username: string;
}

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
  return !!getToken();
}

export function getAuthHeaders(): HeadersInit {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function login(username: string, password: string): Promise<AuthResponse> {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Login failed' }));
    throw new Error(error.detail || 'Login failed');
  }

  const data: AuthResponse = await response.json();
  setToken(data.access_token);
  return data;
}

export function logout(): void {
  removeToken();
}

export function parseToken(token: string): User | null {
  try {
    const payload = token.split('.')[1];
    const decoded = JSON.parse(atob(payload));
    return { username: decoded.sub };
  } catch {
    return null;
  }
}

export function getCurrentUser(): User | null {
  const token = getToken();
  if (!token) return null;
  return parseToken(token);
}
