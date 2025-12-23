import { motion } from 'framer-motion';
import { Shield, Cookie, Server, Eye, Globe, CreditCard, Scale, AlertTriangle } from 'lucide-react';
import { useObfuscatedContact } from '@/lib/obfuscate';

export function PrivacyPage() {
  const { decodedEmail, decode, decoded } = useObfuscatedContact();

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-3xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Datenschutzerklärung</h1>
        <p className="text-muted-foreground mt-1">Privacy Policy (GDPR / DSGVO)</p>
        <p className="text-xs text-muted-foreground mt-2">
          This is a personal, non-commercial project. No fees are charged for using this service.
        </p>
      </div>

      {/* 1. Controller Information */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">1. Data Controller (Verantwortlicher)</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>The controller responsible for data processing on this website is:</p>
          <div className="bg-muted/50 p-4 rounded-lg space-y-1">
            <p className="font-medium text-foreground">Karl Machleidt</p>
            {/* 
              ⚠️ IMPORTANT: You MUST add your real postal address here.
              This is required by GDPR Art. 13(1)(a) and TMG § 5.
              Example:
              <p>Musterstraße 123</p>
              <p>12345 Berlin, Germany</p>
            */}
            <p className="text-warning">[Postal address required]</p>
            <p>
              E-Mail:{' '}
              {decoded ? (
                <a href={`mailto:${decodedEmail}`} className="text-primary underline">
                  {decodedEmail}
                </a>
              ) : (
                <button onClick={decode} className="text-primary underline">
                  Click to reveal
                </button>
              )}
            </p>
          </div>
          <p>
            <strong className="text-foreground">Data Protection Officer:</strong> Not appointed 
            (not required under GDPR Art. 37 for private operators without large-scale processing).
          </p>
        </div>
      </section>

      {/* 2. Overview */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">2. Privacy at a Glance</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            This website collects and processes personal data only to the minimum extent necessary 
            for its operation. We do not sell your data, do not use advertising trackers, and 
            employ privacy-preserving analytics.
          </p>
          <div className="grid sm:grid-cols-2 gap-3">
            <div className="bg-green-500/10 border border-green-500/20 p-3 rounded-lg">
              <p className="font-medium text-green-600 dark:text-green-400">✓ No advertising cookies</p>
            </div>
            <div className="bg-green-500/10 border border-green-500/20 p-3 rounded-lg">
              <p className="font-medium text-green-600 dark:text-green-400">✓ No data selling</p>
            </div>
            <div className="bg-green-500/10 border border-green-500/20 p-3 rounded-lg">
              <p className="font-medium text-green-600 dark:text-green-400">✓ Self-hosted analytics</p>
            </div>
            <div className="bg-green-500/10 border border-green-500/20 p-3 rounded-lg">
              <p className="font-medium text-green-600 dark:text-green-400">✓ Minimal data collection</p>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Hosting & Server Logs */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Server className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">3. Hosting & Server Logs</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            This website is self-hosted on private infrastructure. When you visit this website, 
            your browser automatically transmits technical data that is logged for security and 
            operational purposes:
          </p>
          <ul className="list-disc pl-6 space-y-1">
            <li>IP address (anonymized after processing)</li>
            <li>Date and time of access</li>
            <li>Requested URL and referrer</li>
            <li>Browser type and operating system</li>
            <li>HTTP status code</li>
          </ul>
          <div className="bg-muted/50 p-3 rounded-lg space-y-2">
            <p><strong className="text-foreground">Legal Basis:</strong> Article 6(1)(f) GDPR 
            (legitimate interest in the secure and stable operation of the website).</p>
            <p><strong className="text-foreground">Retention:</strong> Server logs are automatically 
            deleted after 7 days.</p>
          </div>
        </div>
      </section>

      {/* 4. Cloudflare CDN */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Globe className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">4. Content Delivery & Security (Cloudflare)</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            We use Cloudflare, Inc. as a content delivery network (CDN) and security service. 
            All traffic to this website passes through Cloudflare's global network, which provides:
          </p>
          <ul className="list-disc pl-6 space-y-1">
            <li>DDoS protection and web application firewall</li>
            <li>TLS/SSL encryption</li>
            <li>Content caching for faster load times</li>
          </ul>
          <p>Cloudflare may process the following data:</p>
          <ul className="list-disc pl-6 space-y-1">
            <li>IP addresses</li>
            <li>Security-related request metadata</li>
            <li>Performance metrics</li>
          </ul>
          <div className="bg-warning/10 border border-warning/20 p-3 rounded-lg space-y-2">
            <p><strong className="text-foreground">Third-Country Transfer:</strong> Cloudflare, Inc. 
            is a US company. Data transfers to the US are protected by EU Standard Contractual 
            Clauses (SCCs) under GDPR Art. 46(2)(c).</p>
            <p><strong className="text-foreground">Role:</strong> Cloudflare acts as a data processor 
            on our behalf under a Data Processing Agreement.</p>
          </div>
          <p>
            More information:{' '}
            <a 
              href="https://www.cloudflare.com/privacypolicy/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-4"
            >
              Cloudflare Privacy Policy
            </a>
          </p>
          <p>
            <strong className="text-foreground">Legal Basis:</strong> Article 6(1)(f) GDPR 
            (legitimate interest in website security and performance).
          </p>
        </div>
      </section>

      {/* 5. Analytics */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Eye className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">5. Web Analytics (Umami)</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            We use Umami, a privacy-focused, open-source analytics platform that is <strong>self-hosted 
            on our own infrastructure</strong>. No data is sent to third parties.
          </p>
          <div className="bg-green-500/10 border border-green-500/20 p-3 rounded-lg">
            <p className="font-medium text-green-600 dark:text-green-400">
              Umami is configured for maximum privacy:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-1 text-muted-foreground">
              <li>No cookies are set</li>
              <li>IP addresses are not stored</li>
              <li>No personal identifiers are collected</li>
              <li>DNT (Do Not Track) header is respected</li>
            </ul>
          </div>
          <p><strong className="text-foreground">Collected data (aggregated, anonymous):</strong></p>
          <ul className="list-disc pl-6 space-y-1">
            <li>Page views and session duration</li>
            <li>Referrer (where you came from)</li>
            <li>Device type and browser (generic)</li>
            <li>Country (derived from IP, not stored)</li>
          </ul>
          <p>
            <strong className="text-foreground">Legal Basis:</strong> Article 6(1)(f) GDPR 
            (legitimate interest in understanding website usage to improve the service).
          </p>
          <p>
            <strong className="text-foreground">Opt-Out:</strong> Enable "Do Not Track" in your 
            browser settings, and analytics will be disabled for your visits.
          </p>
        </div>
      </section>

      {/* 6. Cookies & Local Storage */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Cookie className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">6. Cookies & Browser Storage</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>This website uses the following storage mechanisms:</p>
          
          <h3 className="font-medium text-foreground pt-2">Strictly Necessary (No Consent Required)</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b">
                  <th className="p-2">Name</th>
                  <th className="p-2">Purpose</th>
                  <th className="p-2">Duration</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-2 font-mono text-xs">session</td>
                  <td className="p-2">Authentication (login state)</td>
                  <td className="p-2">Session / 7 days</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2 font-mono text-xs">theme</td>
                  <td className="p-2">Light/dark mode preference</td>
                  <td className="p-2">Persistent</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="font-medium text-foreground pt-4">Functional / Anti-Fraud</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b">
                  <th className="p-2">Name</th>
                  <th className="p-2">Purpose</th>
                  <th className="p-2">Duration</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-2 font-mono text-xs">dip_device_id</td>
                  <td className="p-2">Prevents vote manipulation (one vote per device per stock)</td>
                  <td className="p-2">2 years</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 p-3 rounded-lg">
            <p>
              <strong className="text-foreground">Note on Anti-Fraud Identifier:</strong> We store 
              a random device identifier to prevent vote manipulation. This identifier is a random 
              string that cannot be used to identify you personally. It is stored in cookies and 
              browser storage (localStorage, IndexedDB) to maintain consistency across sessions.
            </p>
            <p className="mt-2">
              <strong className="text-foreground">Legal Basis:</strong> Article 6(1)(f) GDPR 
              (legitimate interest in preventing abuse and ensuring fair voting). This processing 
              is strictly necessary for the integrity of the voting system (§ 25 Abs. 2 Nr. 2 TTDSG).
            </p>
            <p className="mt-2">
              <strong className="text-foreground">Note:</strong> This identifier is essential for 
              the voting functionality and cannot be disabled. Clearing your browser data will reset 
              the identifier but also your voting history.
            </p>
          </div>
        </div>
      </section>

      {/* 7. PayPal Donations */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <CreditCard className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">7. Donations (PayPal)</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            This website offers the option to make voluntary donations via PayPal. Donations are 
            entirely optional and not required to use any feature of this website.
          </p>
          <p>
            When you click the donation link, you are redirected to PayPal.com. We do not load 
            any PayPal scripts or set any PayPal cookies on this website.
          </p>
          <p>
            <strong className="text-foreground">Data processed by PayPal:</strong> When you complete 
            a donation, PayPal processes your payment information (name, email, payment details) 
            as an <strong>independent data controller</strong>. We only receive your name and email 
            for correspondence purposes.
          </p>
          <div className="bg-warning/10 border border-warning/20 p-3 rounded-lg">
            <p><strong className="text-foreground">Third-Country Transfer:</strong> PayPal 
            (Europe) S.à r.l. et Cie, S.C.A. may transfer data to PayPal, Inc. in the US under 
            Standard Contractual Clauses.</p>
          </div>
          <p>
            PayPal Privacy Policy:{' '}
            <a 
              href="https://www.paypal.com/webapps/mpp/ua/privacy-full" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-4"
            >
              paypal.com/privacy
            </a>
          </p>
          <p>
            <strong className="text-foreground">Legal Basis:</strong> Article 6(1)(b) GDPR 
            (performance of a contract – your voluntary donation).
          </p>
        </div>
      </section>

      {/* 8. Your Rights */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Scale className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">8. Your Rights (GDPR)</h2>
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>Under the General Data Protection Regulation, you have the following rights:</p>
          <ul className="list-disc pl-6 space-y-2">
            <li><strong className="text-foreground">Right of Access (Art. 15)</strong> – Request information about your stored data</li>
            <li><strong className="text-foreground">Right to Rectification (Art. 16)</strong> – Request correction of inaccurate data</li>
            <li><strong className="text-foreground">Right to Erasure (Art. 17)</strong> – Request deletion of your data ("right to be forgotten")</li>
            <li><strong className="text-foreground">Right to Restriction (Art. 18)</strong> – Request restriction of processing</li>
            <li><strong className="text-foreground">Right to Data Portability (Art. 20)</strong> – Request your data in a machine-readable format</li>
            <li><strong className="text-foreground">Right to Object (Art. 21)</strong> – Object to processing based on legitimate interest</li>
          </ul>
          <p>To exercise any of these rights, please contact us at the email address above.</p>
          
          <h3 className="font-medium text-foreground pt-4">Right to Lodge a Complaint</h3>
          <p>
            You have the right to lodge a complaint with a supervisory authority if you believe 
            that the processing of your personal data violates the GDPR (Art. 77 GDPR).
          </p>
          <div className="bg-muted/50 p-3 rounded-lg">
            <p><strong className="text-foreground">Competent Supervisory Authority:</strong></p>
            {/* Replace with YOUR state's DPA */}
            <p className="mt-1">[Your state's data protection authority]</p>
            <p className="mt-1">
              List of German DPAs:{' '}
              <a 
                href="https://www.bfdi.bund.de/DE/Service/Anschriften/Laender/Laender-node.html" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary underline"
              >
                bfdi.bund.de
              </a>
            </p>
          </div>
        </div>
      </section>

      {/* 9. Data Security */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">9. Data Security</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>We implement appropriate technical and organizational measures to protect your data:</p>
          <ul className="list-disc pl-6 space-y-1">
            <li>TLS 1.3 encryption for all connections (HTTPS)</li>
            <li>HTTP Strict Transport Security (HSTS)</li>
            <li>Regular security updates and monitoring</li>
            <li>Access controls and authentication</li>
            <li>Cloudflare DDoS protection and Web Application Firewall</li>
          </ul>
        </div>
      </section>

      {/* 10. Changes */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">10. Changes to This Policy</h2>
        <p className="text-sm text-muted-foreground">
          We may update this privacy policy from time to time. Changes will be posted on this page 
          with an updated revision date.
        </p>
      </section>

      {/* Not Financial Advice */}
      <section className="space-y-4 p-4 rounded-xl bg-warning/10 border border-warning/20">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-warning" />
          <h2 className="text-xl font-semibold text-warning">Not Financial Advice</h2>
        </div>
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
