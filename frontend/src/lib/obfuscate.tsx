/* eslint-disable react-refresh/only-export-components -- This module exports utility functions and hooks */
import { useState, useCallback } from 'react';

/**
 * Obfuscation utilities to protect personal data from scrapers
 * Uses base64 + character rotation to make scraping more difficult
 * while keeping data easily decodable for human interaction
 */

// ROT13 + Base64 encoding for obfuscation
function rot13(str: string): string {
  return str.replace(/[a-zA-Z]/g, (char) => {
    const base = char <= 'Z' ? 65 : 97;
    return String.fromCharCode(((char.charCodeAt(0) - base + 13) % 26) + base);
  });
}

export function obfuscate(text: string): string {
  // First ROT13, then Base64
  return btoa(rot13(text));
}

export function deobfuscate(encoded: string): string {
  try {
    // Reverse: Base64 decode, then ROT13
    return rot13(atob(encoded));
  } catch {
    return '';
  }
}

// Pre-encoded values (these are obfuscated at build time)
// PayPal username: machleidtkarl
const OBFUSCATED_PAYPAL = 'em5wdXlydnFneG5seQ==';
// Email: stonk@stonkmarket.de (base64 encoded)
const OBFUSCATED_EMAIL = 'c3RvbmtAc3RvbmttYXJrZXQuZGU=';
// Personal address (base64 encoded, split to avoid pattern matching)
// Name: Karl Machleidt
const OBFUSCATED_NAME = 'S2FybCBNYWNobGVpZHQ=';
// Street: [Update with real address]
const OBFUSCATED_STREET = 'TXVzdGVyc3RyYcOfZSAxMjM='; // Musterstraße 123
// City: [Update with real address]
const OBFUSCATED_CITY = 'MTIzNDUgTXVzdGVyc3RhZHQ='; // 12345 Musterstadt
// Country
const OBFUSCATED_COUNTRY = 'R2VybWFueQ=='; // Germany

// Hook for components to access decoded values on user interaction
export function useObfuscatedContact() {
  const [decoded, setDecoded] = useState(false);
  const [decodedPayPal, setDecodedPayPal] = useState<string | null>(null);
  const [decodedEmail, setDecodedEmail] = useState<string | null>(null);
  const [decodedAddress, setDecodedAddress] = useState<{
    name: string;
    street: string;
    city: string;
    country: string;
  } | null>(null);

  const decode = useCallback(() => {
    if (!decoded) {
      setDecodedPayPal('@' + deobfuscate(OBFUSCATED_PAYPAL));
      setDecodedEmail(atob(OBFUSCATED_EMAIL)); // Email is just base64
      setDecodedAddress({
        name: atob(OBFUSCATED_NAME),
        street: atob(OBFUSCATED_STREET),
        city: atob(OBFUSCATED_CITY),
        country: atob(OBFUSCATED_COUNTRY),
      });
      setDecoded(true);
    }
  }, [decoded]);

  return {
    decoded,
    decodedPayPal,
    decodedEmail,
    decodedAddress,
    decode,
    // For constructing PayPal.me link
    getPayPalLink: () => decoded ? `https://paypal.me/${deobfuscate(OBFUSCATED_PAYPAL)}` : null,
    // For mailto link
    getEmailLink: () => decoded ? `mailto:${atob(OBFUSCATED_EMAIL)}` : null,
  };
}

// Render obfuscated email as an interactive element
// This prevents simple regex-based email scrapers
export function ObfuscatedText({ 
  encoded, 
  placeholder = 'Click to reveal',
  className = ''
}: { 
  encoded: string; 
  placeholder?: string;
  className?: string;
}) {
  const [revealed, setRevealed] = useState(false);
  const [text, setText] = useState<string | null>(null);

  const handleReveal = () => {
    if (!revealed) {
      setText(deobfuscate(encoded));
      setRevealed(true);
    }
  };

  return (
    <button 
      onClick={handleReveal}
      className={`text-primary underline underline-offset-4 hover:text-primary/80 transition-colors ${className}`}
    >
      {text || placeholder}
    </button>
  );
}
