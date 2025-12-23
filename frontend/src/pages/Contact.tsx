import { motion } from 'framer-motion';
import { Mail, Heart, MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useObfuscatedContact } from '@/lib/obfuscate';
import { useSEO, generateBreadcrumbJsonLd } from '@/lib/seo';

export function ContactPage() {
  const { decoded, decode, decodedPayPal, getPayPalLink } = useObfuscatedContact();

  // SEO for Contact page
  useSEO({
    title: 'Contact Us',
    description: 'Get in touch with the StonkMarket team. We welcome feedback, bug reports, and feature suggestions.',
    keywords: 'contact, feedback, support, bug report, feature request',
    canonical: '/contact',
    jsonLd: generateBreadcrumbJsonLd([
      { name: 'Home', url: '/' },
      { name: 'Contact', url: '/contact' },
    ]),
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl mx-auto space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Contact</h1>
        <p className="text-muted-foreground mt-1">Get in touch with us</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageCircle className="h-5 w-5" />
            How to Contact Us
          </CardTitle>
          <CardDescription>
            We'd love to hear from you
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <motion.div 
            className="p-6 bg-muted/50 rounded-xl space-y-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
          >
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Mail className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold">Send us a message via PayPal</h3>
                <p className="text-sm text-muted-foreground">
                  The easiest way to reach us is through PayPal's messaging system
                </p>
              </div>
            </div>

            <div className="space-y-3 text-sm">
              <p>To contact us:</p>
              <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
                <li>Click the button below to open PayPal</li>
                <li>Enter any amount (even $0.01)</li>
                <li>Use the "Add a note" field to write your message</li>
                <li>We'll respond via PayPal's messaging system</li>
              </ol>
            </div>

            {decoded ? (
              <Button 
                asChild 
                size="lg" 
                className="w-full"
              >
                <a 
                  href={getPayPalLink() || '#'} 
                  target="_blank" 
                  rel="noopener noreferrer"
                >
                  <Mail className="h-4 w-4 mr-2" />
                  Message via PayPal ({decodedPayPal})
                </a>
              </Button>
            ) : (
              <Button 
                size="lg" 
                className="w-full"
                onClick={decode}
              >
                <Mail className="h-4 w-4 mr-2" />
                Reveal Contact Link
              </Button>
            )}
          </motion.div>

          <motion.div 
            className="p-6 bg-primary/5 rounded-xl border border-primary/20 space-y-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
          >
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Heart className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold">Support the Project</h3>
                <p className="text-sm text-muted-foreground">
                  If you find StonkMarket useful, consider supporting its development
                </p>
              </div>
            </div>

            {decoded ? (
              <Button 
                asChild 
                variant="outline"
                size="lg" 
                className="w-full"
              >
                <a 
                  href={getPayPalLink() || '#'} 
                  target="_blank" 
                  rel="noopener noreferrer"
                >
                  <Heart className="h-4 w-4 mr-2" />
                  Donate via PayPal
                </a>
              </Button>
            ) : (
              <Button 
                variant="outline"
                size="lg" 
                className="w-full"
                onClick={decode}
              >
                <Heart className="h-4 w-4 mr-2" />
                Show Donation Link
              </Button>
            )}
          </motion.div>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground text-center">
        We typically respond within 1-2 business days
      </p>
    </motion.div>
  );
}
