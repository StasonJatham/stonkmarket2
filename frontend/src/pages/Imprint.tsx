import { motion } from 'framer-motion';
import { useObfuscatedContact } from '@/lib/obfuscate';

export function ImprintPage() {
  const { decodedEmail, decode, decoded } = useObfuscatedContact();

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Legal Notice</h1>
        <p className="text-muted-foreground mt-1">Imprint (§5 TMG / §18 MStV)</p>
      </div>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Information according to § 5 TMG</h2>
        <div className="space-y-2 text-muted-foreground">
          <p><strong className="text-foreground">Responsible for content:</strong></p>
          <p>Karl Machleidt</p>
          <p>
            <strong className="text-foreground">Contact:</strong>{' '}
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
                Click to reveal
              </button>
            )}
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Disclaimer</h2>
        
        <div className="space-y-3 text-sm text-muted-foreground">
          <h3 className="font-medium text-foreground">Liability for Content</h3>
          <p>
            The contents of our pages have been created with the utmost care. However, we cannot 
            guarantee the accuracy, completeness, and timeliness of the content.
          </p>
          
          <h3 className="font-medium text-foreground pt-2">Liability for Links</h3>
          <p>
            Our website contains links to external websites of third parties, over whose content 
            we have no influence. The respective provider is always responsible for the content 
            of the linked pages.
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Copyright</h2>
        <p className="text-sm text-muted-foreground">
          The content and works created by the site operators on these pages are subject to 
          copyright law. Reproduction, editing, distribution, and any kind of exploitation 
          outside the limits of copyright require the written consent of the respective 
          author or creator.
        </p>
      </section>

      <section className="space-y-4 p-4 rounded-xl bg-warning/10 border border-warning/20">
        <h2 className="text-xl font-semibold text-warning">Not Financial Advice</h2>
        <p className="text-sm text-muted-foreground">
          The information presented on this website does <strong>not constitute financial advice</strong> 
          and should not be understood as such. All content is for general information and 
          entertainment purposes only. Use of the provided information is at your own risk. 
          Before making any investment decision, you should consult a qualified financial advisor.
        </p>
      </section>

      <p className="text-xs text-muted-foreground pt-4 border-t">
        Last updated: December 2025
      </p>
    </motion.div>
  );
}
