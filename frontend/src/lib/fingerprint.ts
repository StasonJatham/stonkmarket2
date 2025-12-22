/**
 * Multi-layered client fingerprinting for abuse prevention.
 *
 * Stores a persistent device ID across multiple storage mechanisms:
 * - Cookies
 * - localStorage
 * - sessionStorage
 * - IndexedDB
 *
 * If any storage has an existing ID, it syncs to all others.
 * This makes it harder for users to game the voting system.
 */

const FINGERPRINT_KEY = 'dip_device_id';
const FINGERPRINT_VERSION = 1;
const DB_NAME = 'dipfinder_db';
const STORE_NAME = 'fingerprints';

/**
 * Generate a random device ID
 */
function generateDeviceId(): string {
  const timestamp = Date.now().toString(36);
  const randomPart = Math.random().toString(36).substring(2, 15);
  const randomPart2 = Math.random().toString(36).substring(2, 15);
  return `v${FINGERPRINT_VERSION}_${timestamp}_${randomPart}${randomPart2}`;
}

/**
 * Get value from cookie
 */
function getCookie(name: string): string | null {
  const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
  return match ? decodeURIComponent(match[2]) : null;
}

/**
 * Set cookie with long expiry (2 years)
 */
function setCookie(name: string, value: string): void {
  const maxAge = 2 * 365 * 24 * 60 * 60; // 2 years in seconds
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; max-age=${maxAge}; SameSite=Lax`;
}

/**
 * Get value from localStorage
 */
function getLocalStorage(): string | null {
  try {
    return localStorage.getItem(FINGERPRINT_KEY);
  } catch {
    return null;
  }
}

/**
 * Set value in localStorage
 */
function setLocalStorage(value: string): void {
  try {
    localStorage.setItem(FINGERPRINT_KEY, value);
  } catch {
    // Ignore - localStorage may be disabled
  }
}

/**
 * Get value from sessionStorage
 */
function getSessionStorage(): string | null {
  try {
    return sessionStorage.getItem(FINGERPRINT_KEY);
  } catch {
    return null;
  }
}

/**
 * Set value in sessionStorage
 */
function setSessionStorage(value: string): void {
  try {
    sessionStorage.setItem(FINGERPRINT_KEY, value);
  } catch {
    // Ignore
  }
}

/**
 * Open IndexedDB database
 */
function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'key' });
      }
    };
  });
}

/**
 * Get value from IndexedDB
 */
async function getIndexedDB(): Promise<string | null> {
  try {
    const db = await openDatabase();
    return new Promise((resolve) => {
      const transaction = db.transaction(STORE_NAME, 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(FINGERPRINT_KEY);

      request.onsuccess = () => {
        resolve(request.result?.value || null);
      };
      request.onerror = () => resolve(null);

      transaction.oncomplete = () => db.close();
    });
  } catch {
    return null;
  }
}

/**
 * Set value in IndexedDB
 */
async function setIndexedDB(value: string): Promise<void> {
  try {
    const db = await openDatabase();
    return new Promise((resolve) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      store.put({ key: FINGERPRINT_KEY, value, updatedAt: Date.now() });

      transaction.oncomplete = () => {
        db.close();
        resolve();
      };
      transaction.onerror = () => {
        db.close();
        resolve();
      };
    });
  } catch {
    // Ignore
  }
}

/**
 * Sync device ID to all storage mechanisms
 */
async function syncToAllStorages(deviceId: string): Promise<void> {
  setCookie(FINGERPRINT_KEY, deviceId);
  setLocalStorage(deviceId);
  setSessionStorage(deviceId);
  await setIndexedDB(deviceId);
}

/**
 * Get the device fingerprint.
 *
 * Checks all storage mechanisms and returns the first found ID.
 * If any storage has an ID, syncs it to all others.
 * If no ID exists, generates a new one and stores everywhere.
 *
 * This is a synchronous fast path with async IndexedDB check.
 */
export async function getDeviceFingerprint(): Promise<string> {
  // Fast synchronous checks first
  const cookie = getCookie(FINGERPRINT_KEY);
  const local = getLocalStorage();
  const session = getSessionStorage();

  // If we have any sync value, use it and sync to others in background
  const existingSync = cookie || local || session;

  // Check IndexedDB (async)
  const idb = await getIndexedDB();

  // Find the first existing ID
  const existingId = existingSync || idb;

  if (existingId) {
    // Sync to all storages in background (don't await)
    syncToAllStorages(existingId);
    return existingId;
  }

  // No existing ID - generate new one
  const newId = generateDeviceId();
  await syncToAllStorages(newId);
  return newId;
}

/**
 * Get device fingerprint synchronously (fast path).
 * Returns null if no sync storage has a value yet.
 * Useful for quick checks without waiting for IndexedDB.
 */
export function getDeviceFingerprintSync(): string | null {
  return getCookie(FINGERPRINT_KEY) || getLocalStorage() || getSessionStorage();
}

/**
 * Check if this browser has voted on a specific suggestion.
 * Tracked separately from the device fingerprint.
 */
export function hasVotedLocally(symbol: string): boolean {
  try {
    const votes = JSON.parse(localStorage.getItem('dip_votes') || '{}');
    return !!votes[symbol.toUpperCase()];
  } catch {
    return false;
  }
}

/**
 * Record that this browser voted on a suggestion.
 */
export function recordLocalVote(symbol: string): void {
  try {
    const votes = JSON.parse(localStorage.getItem('dip_votes') || '{}');
    votes[symbol.toUpperCase()] = Date.now();
    localStorage.setItem('dip_votes', JSON.stringify(votes));
  } catch {
    // Ignore
  }
}

/**
 * Get all locally recorded votes.
 */
export function getLocalVotes(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem('dip_votes') || '{}');
  } catch {
    return {};
  }
}
