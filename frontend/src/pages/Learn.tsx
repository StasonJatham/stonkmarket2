import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  TrendingDown, 
  AlertTriangle,
  BookOpen,
  Target,
  Clock,
  BarChart3,
  Brain,
  Shield,
  ArrowRight,
  Lightbulb,
  CheckCircle2,
  XCircle,
  Zap,
} from 'lucide-react';
import { useSEO, generateBreadcrumbJsonLd, generateFAQJsonLd } from '@/lib/seo';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

// Feature flags from environment
const ENABLE_LEGAL_PAGES = import.meta.env.VITE_ENABLE_LEGAL_PAGES === 'true';
import { Badge } from '@/components/ui/badge';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: [0, 0, 0.2, 1] as const,
    },
  },
};

const fadeInVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5 },
  },
};

// Educational content structured as articles for SEO
const ARTICLES = [
  {
    id: 'what-is-a-dip',
    title: 'What is a Stock Dip?',
    icon: TrendingDown,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    summary: 'Understanding price pullbacks and their significance in the market.',
    content: `A stock dip occurs when a stock's price falls significantly below its recent high point. This is typically measured from the 52-week high or a shorter reference period. For example, if a stock was trading at $100 and drops to $80, that's a 20% dip.

Dips can happen for various reasons: market-wide corrections, sector rotation, company-specific news, or simply profit-taking by investors. Not all dips are equal—some represent buying opportunities while others signal fundamental problems with the company.`,
  },
  {
    id: 'buy-the-dip-strategy',
    title: 'The "Buy the Dip" Strategy',
    icon: Target,
    color: 'text-chart-1',
    bgColor: 'bg-chart-1/10',
    summary: 'How investors use price pullbacks as potential entry points.',
    content: `"Buy the dip" is an investment strategy where investors purchase stocks after a significant price decline, betting on a recovery. The logic is simple: if a fundamentally strong company's stock drops due to temporary factors, buying at the lower price could yield profits when the stock recovers.

However, this strategy requires careful analysis. You need to distinguish between temporary pullbacks in quality stocks and genuine downtrends in declining businesses. Key factors to consider include the company's fundamentals, the reason for the dip, and broader market conditions.`,
  },
  {
    id: 'identifying-quality-dips',
    title: 'Identifying Quality Dip Opportunities',
    icon: BarChart3,
    color: 'text-purple-500',
    bgColor: 'bg-purple-500/10',
    summary: 'Criteria for finding stocks with genuine recovery potential.',
    content: `Not every dip is worth buying. Here's what to look for:

**Strong Fundamentals**: The company should have solid revenue growth, healthy profit margins, and a sustainable business model.

**Temporary Catalyst**: The dip should be caused by temporary factors (market sentiment, short-term news) rather than fundamental deterioration.

**Historical Recovery**: Check if the stock has recovered from similar dips in the past.

**Valuation**: Even after the dip, ensure the stock isn't overvalued compared to peers.

**Volume Analysis**: Look for signs of institutional buying during the dip.`,
  },
  {
    id: 'risks-of-dip-buying',
    title: 'Risks of Buying Dips',
    icon: AlertTriangle,
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/10',
    summary: 'Understanding the dangers and how to protect yourself.',
    content: `Buying dips carries significant risks that every investor should understand:

**Catching a Falling Knife**: Sometimes a stock dips and keeps falling. What looks like a 20% discount today could become a 50% loss tomorrow.

**Value Traps**: Some stocks are cheap for good reasons. Declining businesses rarely recover.

**Opportunity Cost**: Money tied up in underperforming dip-buys could be invested elsewhere.

**Emotional Decision-Making**: The fear of missing out (FOMO) can lead to impulsive purchases.

**Timing Risk**: Even good companies can take years to recover from significant dips.`,
  },
  {
    id: 'using-stonkmarket',
    title: 'How StonkMarket Helps',
    icon: Brain,
    color: 'text-primary',
    bgColor: 'bg-primary/10',
    summary: 'Using our tools to research and track potential opportunities.',
    content: `StonkMarket provides several tools to help with your dip research:

**Dip Tracker**: Our dashboard shows stocks currently trading below their reference highs, sorted by dip depth and other metrics.

**DipSwipe**: A Tinder-style interface to quickly review stocks with AI-generated summaries.

**AI Analysis**: GPT-powered insights providing context, ratings, and reasoning for each stock.

**Community Voting**: See what other users think about specific dip opportunities.

**Benchmark Comparison**: Compare individual stocks against S&P 500 and MSCI World indices.

Remember: Our tools are for research and education only. Always do your own due diligence before making any investment decisions.`,
  },
];

// Tips & Tricks
const TIPS = [
  {
    type: 'do',
    icon: CheckCircle2,
    title: 'Do: Diversify Your Dips',
    description: "Don't put all your money into one dipped stock. Spread across multiple positions to reduce risk.",
  },
  {
    type: 'do',
    icon: CheckCircle2,
    title: 'Do: Wait for Stabilization',
    description: "Let a stock show signs of stabilization before buying. Patience often pays off.",
  },
  {
    type: 'do',
    icon: CheckCircle2,
    title: 'Do: Set Price Alerts',
    description: "Use price alerts to notify you when stocks hit your target entry points.",
  },
  {
    type: 'dont',
    icon: XCircle,
    title: "Don't: Chase Every Dip",
    description: "Not every dip is an opportunity. Be selective and do your research first.",
  },
  {
    type: 'dont',
    icon: XCircle,
    title: "Don't: Ignore the Why",
    description: "Always understand why a stock dropped. News matters—a lot.",
  },
  {
    type: 'dont',
    icon: XCircle,
    title: "Don't: Forget Stop-Losses",
    description: "Have an exit plan. Know when to cut losses if your thesis is wrong.",
  },
];

// Quick tips for sidebar
const QUICK_TIPS = [
  "A 10-20% drop is typically considered a 'dip'",
  "Check the VIX (fear index) before buying dips",
  "Earnings season often creates dip opportunities",
  "Dollar-cost averaging reduces timing risk",
  "Blue-chip stocks recover faster than small caps",
];

// FAQs for structured data
const FAQS = [
  {
    question: 'Is buying the dip a good strategy?',
    answer: 'Buying the dip can be profitable if applied to fundamentally strong companies experiencing temporary setbacks. However, it carries risks including catching falling knives and value traps. Success requires thorough research, patience, and proper risk management.',
  },
  {
    question: 'How do I know if a dip is a buying opportunity?',
    answer: 'Look for stocks with strong fundamentals (revenue growth, profit margins, competitive advantage) where the dip was caused by temporary factors rather than deteriorating business conditions. Check historical recovery patterns and ensure the valuation is reasonable even after the decline.',
  },
  {
    question: 'What percentage drop counts as a dip?',
    answer: 'There is no universal definition, but many investors consider a 10-20% decline from recent highs as a "dip." Larger declines (20-40%) are often called corrections, while drops over 40-50% may indicate more serious problems. StonkMarket tracks stocks with dips of 10% or more.',
  },
  {
    question: 'Should I use stop-loss orders when buying dips?',
    answer: 'Stop-loss orders can help limit downside risk, but they may also trigger during temporary volatility, locking in losses. Consider your time horizon, risk tolerance, and the specific situation. Some investors prefer position sizing and diversification over stop-losses for dip-buying strategies.',
  },
  {
    question: 'How long does it take for stocks to recover from dips?',
    answer: 'Recovery time varies significantly depending on the cause of the dip, market conditions, and company fundamentals. Some stocks recover within weeks, while others take months or years. Historical data shows that major market indices typically recover from corrections within 6-18 months, but individual stocks vary widely.',
  },
  {
    question: 'What is a value trap?',
    answer: 'A value trap is a stock that appears cheap based on valuation metrics but continues to decline. These are often companies with deteriorating fundamentals, disrupted business models, or structural problems. The key is distinguishing between temporary dips and permanent declines.',
  },
  {
    question: 'How much of my portfolio should I allocate to dip buying?',
    answer: 'Most financial advisors suggest keeping 5-15% of your portfolio as "opportunity cash" for dip buying. Never invest more than you can afford to lose, and ensure your core portfolio is properly diversified before attempting dip-buying strategies.',
  },
];

// Generate Article schema for SEO
function generateArticleJsonLd(article: typeof ARTICLES[0]) {
  return {
    '@context': 'https://schema.org',
    '@type': 'Article',
    'headline': article.title,
    'description': article.summary,
    'author': {
      '@type': 'Organization',
      'name': 'StonkMarket',
      'url': 'https://stonkmarket.de',
    },
    'publisher': {
      '@type': 'Organization',
      'name': 'StonkMarket',
      'logo': {
        '@type': 'ImageObject',
        'url': 'https://stonkmarket.de/favicon.svg',
      },
    },
    'mainEntityOfPage': {
      '@type': 'WebPage',
      '@id': `https://stonkmarket.de/learn#${article.id}`,
    },
  };
}

export function LearnPage() {
  // SEO with educational content structured data
  useSEO({
    title: 'Learn - Stock Dip Investing Guide',
    description: 'Learn about stock dips, the buy-the-dip strategy, how to identify quality opportunities, and the risks involved. Free educational resources for investors.',
    keywords: 'stock dips, buy the dip, investing strategy, stock market education, dip buying, value investing, stock analysis',
    canonical: '/learn',
    jsonLd: [
      generateBreadcrumbJsonLd([
        { name: 'Home', url: '/' },
        { name: 'Learn', url: '/learn' },
      ]),
      generateFAQJsonLd(FAQS),
      ...ARTICLES.map(generateArticleJsonLd),
    ],
  });

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <motion.section
        initial="hidden"
        animate="visible"
        variants={fadeInVariants}
        className="text-center py-12 md:py-16"
      >
        <Badge variant="outline" className="mb-4">
          <BookOpen className="mr-1.5 h-3.5 w-3.5" />
          Educational Resources
        </Badge>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
          Stock Dip Investing Guide
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
          Master the fundamentals of dip buying, identify quality opportunities, 
          and understand the risks every smart investor should know.
        </p>
        
        {/* Quick Stats */}
        <div className="flex flex-wrap justify-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4" />
            <span>5 min read</span>
          </div>
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            <span>{ARTICLES.length} topics</span>
          </div>
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            <span>Free forever</span>
          </div>
        </div>
      </motion.section>

      {/* Disclaimer Banner */}
      <motion.section
        initial="hidden"
        animate="visible"
        variants={fadeInVariants}
        className="mb-12"
      >
        <Card className="border-warning/30 bg-warning/5">
          <CardContent className="py-4">
            <div className="flex items-center gap-3">
              <AlertTriangle className="h-5 w-5 text-warning shrink-0" />
              <p className="text-sm text-muted-foreground">
                <span className="font-semibold text-warning">Educational Only:</span>{' '}
                This content does not constitute financial advice. Investing involves risk, 
                including loss of principal. Always consult a qualified financial advisor.
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.section>

      {/* Main Content Grid */}
      <div className="grid gap-12 lg:grid-cols-3 lg:gap-8">
        {/* Articles - Main Column */}
        <motion.section
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="lg:col-span-2 space-y-6"
        >
          {ARTICLES.map((article, index) => {
            const Icon = article.icon;
            return (
              <motion.div key={article.id} variants={itemVariants}>
                <Card id={article.id} className="overflow-hidden hover:shadow-lg transition-shadow">
                  <CardHeader className="pb-4">
                    <div className="flex items-start gap-4">
                      <div className={`flex items-center justify-center w-12 h-12 rounded-xl ${article.bgColor} shrink-0`}>
                        <Icon className={`h-6 w-6 ${article.color}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="secondary" className="text-xs">
                            {index + 1} of {ARTICLES.length}
                          </Badge>
                        </div>
                        <CardTitle className="text-xl">{article.title}</CardTitle>
                        <CardDescription className="mt-1">{article.summary}</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="prose prose-sm dark:prose-invert max-w-none">
                      {article.content.split('\n\n').map((paragraph, i) => {
                        // Handle bold text with **
                        if (paragraph.includes('**')) {
                          const parts = paragraph.split(/\*\*(.*?)\*\*/g);
                          return (
                            <p key={i} className="text-sm text-muted-foreground leading-relaxed mb-3 last:mb-0">
                              {parts.map((part, j) => 
                                j % 2 === 1 ? <strong key={j} className="text-foreground">{part}</strong> : part
                              )}
                            </p>
                          );
                        }
                        return (
                          <p key={i} className="text-sm text-muted-foreground leading-relaxed mb-3 last:mb-0">
                            {paragraph}
                          </p>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </motion.section>

        {/* Sidebar */}
        <motion.aside
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="space-y-6"
        >
          {/* Quick Tips Card */}
          <motion.div variants={itemVariants}>
            <Card className="sticky top-4">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Lightbulb className="h-4 w-4 text-yellow-500" />
                  Quick Tips
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <ul className="space-y-3">
                  {QUICK_TIPS.map((tip, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <Zap className="h-3.5 w-3.5 text-yellow-500 shrink-0 mt-0.5" />
                      <span>{tip}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </motion.div>

          {/* Navigation Card */}
          <motion.div variants={itemVariants}>
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Jump to Section</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <nav className="space-y-1">
                  {ARTICLES.map((article, index) => {
                    const Icon = article.icon;
                    return (
                      <a
                        key={article.id}
                        href={`#${article.id}`}
                        className="flex items-center gap-2 p-2 rounded-md text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                      >
                        <span className="flex items-center justify-center w-5 h-5 rounded-full bg-muted text-xs font-medium">
                          {index + 1}
                        </span>
                        <Icon className="h-3.5 w-3.5" />
                        <span className="truncate">{article.title}</span>
                      </a>
                    );
                  })}
                </nav>
              </CardContent>
            </Card>
          </motion.div>
        </motion.aside>
      </div>

      {/* Tips & Tricks Section */}
      <motion.section
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '-100px' }}
        variants={containerVariants}
        className="mt-16"
      >
        <div className="text-center mb-8">
          <Badge variant="outline" className="mb-3">
            <Lightbulb className="mr-1.5 h-3.5 w-3.5" />
            Pro Tips
          </Badge>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">
            Do's and Don'ts of Dip Buying
          </h2>
          <p className="text-muted-foreground mt-2">
            Practical advice from experienced investors
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {TIPS.map((tip, index) => {
            const Icon = tip.icon;
            const isDo = tip.type === 'do';
            return (
              <motion.div key={index} variants={itemVariants}>
                <Card className={`h-full ${isDo ? 'border-success/30 bg-success/5' : 'border-danger/30 bg-danger/5'}`}>
                  <CardContent className="pt-6">
                    <div className="flex items-start gap-3">
                      <div className={`flex items-center justify-center w-8 h-8 rounded-full shrink-0 ${isDo ? 'bg-success/20' : 'bg-danger/20'}`}>
                        <Icon className={`h-4 w-4 ${isDo ? 'text-success' : 'text-danger'}`} />
                      </div>
                      <div>
                        <h3 className="font-semibold text-sm mb-1">{tip.title}</h3>
                        <p className="text-sm text-muted-foreground">{tip.description}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </motion.section>

      {/* FAQ Section with Accordion */}
      <motion.section
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: '-100px' }}
        variants={fadeInVariants}
        className="mt-16"
      >
        <div className="text-center mb-8">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">
            Frequently Asked Questions
          </h2>
          <p className="text-muted-foreground mt-2">
            Common questions about dip buying strategies
          </p>
        </div>

        <Card className="max-w-3xl mx-auto">
          <CardContent className="pt-6">
            <Accordion type="single" collapsible className="w-full">
              {FAQS.map((faq, index) => (
                <AccordionItem key={index} value={`item-${index}`}>
                  <AccordionTrigger className="text-left hover:no-underline">
                    <span className="font-medium">{faq.question}</span>
                  </AccordionTrigger>
                  <AccordionContent>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {faq.answer}
                    </p>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </CardContent>
        </Card>
      </motion.section>

      {/* CTA Section */}
      <motion.section
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={fadeInVariants}
        className="mt-16"
      >
        <Card className="bg-gradient-to-br from-primary/10 to-primary/5 border-primary/20">
          <CardContent className="py-8">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6 text-center md:text-left">
              <div>
                <h3 className="text-xl font-bold mb-2">Ready to Put It Into Practice?</h3>
                <p className="text-muted-foreground max-w-md">
                  Explore our tools to research stocks currently in a dip and make informed decisions.
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-3">
                <Button asChild size="lg">
                  <Link to="/">
                    View Dashboard
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <Link to="/swipe">
                    Try DipSwipe
                  </Link>
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.section>

      {/* Footer Links */}
      <motion.section
        initial="hidden"
        animate="visible"
        variants={fadeInVariants}
        className="mt-12 mb-8"
      >
        <div className="flex flex-wrap gap-2 justify-center">
          <Button variant="ghost" size="sm" asChild>
            <Link to="/about">About Our Methodology</Link>
          </Button>
          {ENABLE_LEGAL_PAGES && (
            <Button variant="ghost" size="sm" asChild>
              <Link to="/privacy">Privacy Policy</Link>
            </Button>
          )}
          <Button variant="ghost" size="sm" asChild>
            <Link to="/contact">Contact Us</Link>
          </Button>
        </div>
      </motion.section>
    </div>
  );
}
