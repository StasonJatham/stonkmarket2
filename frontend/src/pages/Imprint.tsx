import { motion } from 'framer-motion';
import { useObfuscatedContact } from '@/lib/obfuscate';
import { AlertTriangle } from 'lucide-react';

export function ImprintPage() {
  const { decodedEmail, decode, decoded } = useObfuscatedContact();

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Impressum</h1>
        <p className="text-muted-foreground mt-1">Legal Notice (§ 5 TMG / § 18 MStV)</p>
        <p className="text-xs text-muted-foreground mt-2">
          Personal, non-commercial project – no fees are charged for using this service.
        </p>
      </div>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Information according to § 5 TMG</h2>
        <div className="space-y-2 text-muted-foreground">
          <p><strong className="text-foreground">Responsible for content (Angaben gemäß § 5 TMG):</strong></p>
          <p>Karl Machleidt</p>
          {/* 
            ⚠️ IMPORTANT: You MUST add your real postal address here.
            Example format:
            <p>Musterstraße 123</p>
            <p>12345 Berlin</p>
            <p>Germany</p>
          */}
          <p className="text-warning bg-warning/10 p-2 rounded text-sm">
            [Address must be added here - see Privacy.tsx comments]
          </p>
          <p className="pt-2">
            <strong className="text-foreground">Contact:</strong>
          </p>
          <p>
            E-Mail:{' '}
            {decoded ? (
              <a 
                href={`mailto:${decodedEmail}`}
                className="text-primary underline underline-offset-4 hover:text-primary/80 transition-colors"
              >
                {decodedEmail}
              </a>
            ) : (
              <button 
                onClick={decode}
                className="text-primary underline underline-offset-4 hover:text-primary/80 transition-colors"
              >
                Click to reveal email
              </button>
            )}
          </p>
          <p className="text-sm text-muted-foreground">
            Response time: We aim to respond to inquiries within 24 hours.
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Editorial Responsibility (V.i.S.d.P.)</h2>
        <p className="text-muted-foreground">
          Responsible for editorial content according to § 18 Abs. 2 MStV:
        </p>
        <p className="text-muted-foreground">Karl Machleidt (address as above)</p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Dispute Resolution</h2>
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>
            The European Commission provides a platform for online dispute resolution (OS):{' '}
            <a 
              href="https://ec.europa.eu/consumers/odr/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-4"
            >
              https://ec.europa.eu/consumers/odr/
            </a>
          </p>
          <p>
            We are neither obligated nor willing to participate in dispute resolution 
            proceedings before a consumer arbitration board.
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Disclaimer</h2>
        
        <div className="space-y-3 text-sm text-muted-foreground">
          <h3 className="font-medium text-foreground">Liability for Content</h3>
          <p>
            The contents of our pages have been created with the utmost care. However, we cannot 
            guarantee the accuracy, completeness, and timeliness of the content. As a service 
            provider, we are responsible for our own content on these pages according to § 7 (1) TMG. 
            According to §§ 8 to 10 TMG, however, we are not obligated to monitor transmitted or 
            stored third-party information or to investigate circumstances that indicate illegal activity.
          </p>
          
          <h3 className="font-medium text-foreground pt-2">Liability for Links</h3>
          <p>
            Our website contains links to external websites of third parties, over whose content 
            we have no influence. Therefore, we cannot assume any liability for these external contents. 
            The respective provider or operator of the pages is always responsible for the content 
            of the linked pages.
          </p>
          
          <h3 className="font-medium text-foreground pt-2">Copyright</h3>
          <p>
            The content and works created by the site operators on these pages are subject to 
            German copyright law. Reproduction, editing, distribution, and any kind of exploitation 
            outside the limits of copyright require the written consent of the respective author or creator.
          </p>
        </div>
      </section>

      <section className="space-y-4 p-4 rounded-xl bg-warning/10 border border-warning/20">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-warning" />
          <h2 className="text-xl font-semibold text-warning">No Financial Advice</h2>
        </div>
        <p className="text-sm text-muted-foreground">
          The information presented on this website does <strong>not constitute financial advice</strong> 
          and should not be understood as such. All content is for general information and 
          entertainment purposes only. The use of the provided information is at your own risk. 
          Before making any investment decision, you should consult a qualified financial advisor.
        </p>
      </section>

      <p className="text-xs text-muted-foreground pt-4 border-t">
        Last updated: December 2025
      </p>
    </motion.div>
  );
}
