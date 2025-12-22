import { motion } from 'framer-motion';
import { Shield, Cookie, Server, Eye } from 'lucide-react';

export function PrivacyPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Privacy Policy</h1>
        <p className="text-muted-foreground mt-1">GDPR Compliant</p>
      </div>

      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">1. Privacy at a Glance</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            <strong className="text-foreground">Data Controller:</strong> Karl Machleidt
          </p>
          <p>
            This website collects and processes personal data only to the minimum 
            extent necessary. We respect your privacy and strictly comply with GDPR regulations.
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Server className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">2. Hosting & Server Logs</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            When visiting our website, technical data is automatically collected 
            (e.g., IP address, browser type, access time). This data is necessary 
            for technical operations and is not merged with other data sources.
          </p>
          <p>
            <strong className="text-foreground">Legal Basis:</strong> Article 6(1)(f) GDPR 
            (legitimate interest in the technical provision of the website).
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Eye className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">3. Web Analytics (Umami)</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            We use Umami, a privacy-friendly open-source analytics software. 
            Umami <strong>does not store cookies</strong> and collects <strong>no 
            personal data</strong>. All data is processed anonymously.
          </p>
          <p>
            <strong className="text-foreground">Collected Data:</strong> Page views, 
            referrer, device type, country (based on anonymized IP).
          </p>
          <p>
            <strong className="text-foreground">Legal Basis:</strong> Article 6(1)(f) GDPR 
            (legitimate interest in improving our service).
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Cookie className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">4. Cookies</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            This website uses <strong>only technically necessary cookies</strong> 
            for authentication (login session). No tracking cookies or advertising 
            cookies are used.
          </p>
          <p>
            <strong className="text-foreground">Legal Basis:</strong> Article 6(1)(f) GDPR 
            (necessary for the operation of the website).
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">5. Your Rights</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>You have the right to:</p>
          <ul className="list-disc pl-6 space-y-1">
            <li>Access your stored data (Article 15 GDPR)</li>
            <li>Rectification of inaccurate data (Article 16 GDPR)</li>
            <li>Erasure of your data (Article 17 GDPR)</li>
            <li>Restriction of processing (Article 18 GDPR)</li>
            <li>Data portability (Article 20 GDPR)</li>
            <li>Object to processing (Article 21 GDPR)</li>
          </ul>
          <p>
            To exercise your rights, please contact us via email (see footer).
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">6. External Services</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <h3 className="font-medium text-foreground">Yahoo Finance API</h3>
          <p>
            We use the Yahoo Finance API to display stock prices. 
            No personal data is transmitted to Yahoo.
          </p>
          
          <h3 className="font-medium text-foreground pt-2">PayPal</h3>
          <p>
            We use PayPal for donations. PayPal's privacy policy can be found at:{' '}
            <a 
              href="https://www.paypal.com/webapps/mpp/ua/privacy-full" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-4"
            >
              paypal.com/privacy
            </a>
          </p>
        </div>
      </section>

      <section className="space-y-4 p-4 rounded-xl bg-warning/10 border border-warning/20">
        <h2 className="text-xl font-semibold text-warning">Not Financial Advice</h2>
        <p className="text-sm text-muted-foreground">
          The information presented on this website does <strong>not constitute financial advice</strong>. 
          All stock prices and market analysis are for informational purposes only. 
          Use at your own risk.
        </p>
      </section>

      <p className="text-xs text-muted-foreground pt-4 border-t">
        Last updated: December 2025
      </p>
    </motion.div>
  );
}
