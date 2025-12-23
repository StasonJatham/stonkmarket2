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
  ArrowRight
} from 'lucide-react';
import { useSEO, generateBreadcrumbJsonLd, generateFAQJsonLd } from '@/lib/seo';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

// Educational content structured as articles for SEO
const ARTICLES = [
  {
    id: 'what-is-a-dip',
    title: 'What is a Stock Dip?',
    icon: TrendingDown,
    summary: 'Understanding price pullbacks and their significance in the market.',
    content: `A stock dip occurs when a stock's price falls significantly below its recent high point. 
    This is typically measured from the 52-week high or a shorter reference period. For example, 
    if a stock was trading at $100 and drops to $80, that's a 20% dip.
    
    Dips can happen for various reasons: market-wide corrections, sector rotation, company-specific 
    news, or simply profit-taking by investors. Not all dips are equal—some represent buying 
    opportunities while others signal fundamental problems with the company.`,
  },
  {
    id: 'buy-the-dip-strategy',
    title: 'The "Buy the Dip" Strategy',
    icon: Target,
    summary: 'How investors use price pullbacks as potential entry points.',
    content: `"Buy the dip" is an investment strategy where investors purchase stocks after a 
    significant price decline, betting on a recovery. The logic is simple: if a fundamentally 
    strong company's stock drops due to temporary factors, buying at the lower price could 
    yield profits when the stock recovers.
    
    However, this strategy requires careful analysis. You need to distinguish between temporary 
    pullbacks in quality stocks and genuine downtrends in declining businesses. Key factors to 
    consider include the company's fundamentals, the reason for the dip, and broader market conditions.`,
  },
  {
    id: 'identifying-quality-dips',
    title: 'Identifying Quality Dip Opportunities',
    icon: BarChart3,
    summary: 'Criteria for finding stocks with genuine recovery potential.',
    content: `Not every dip is worth buying. Here's what to look for:
    
    1. **Strong Fundamentals**: The company should have solid revenue growth, healthy profit margins, 
    and a sustainable business model.
    
    2. **Temporary Catalyst**: The dip should be caused by temporary factors (market sentiment, 
    short-term news) rather than fundamental deterioration.
    
    3. **Historical Recovery**: Check if the stock has recovered from similar dips in the past.
    
    4. **Valuation**: Even after the dip, ensure the stock isn't overvalued compared to peers.
    
    5. **Volume Analysis**: Look for signs of institutional buying during the dip.`,
  },
  {
    id: 'risks-of-dip-buying',
    title: 'Risks of Buying Dips',
    icon: AlertTriangle,
    summary: 'Understanding the dangers and how to protect yourself.',
    content: `Buying dips carries significant risks that every investor should understand:
    
    1. **Catching a Falling Knife**: Sometimes a stock dips and keeps falling. What looks like 
    a 20% discount today could become a 50% loss tomorrow.
    
    2. **Value Traps**: Some stocks are cheap for good reasons. Declining businesses rarely recover.
    
    3. **Opportunity Cost**: Money tied up in underperforming dip-buys could be invested elsewhere.
    
    4. **Emotional Decision-Making**: The fear of missing out (FOMO) can lead to impulsive purchases.
    
    5. **Timing Risk**: Even good companies can take years to recover from significant dips.`,
  },
  {
    id: 'using-stonkmarket',
    title: 'How StonkMarket Helps',
    icon: Brain,
    summary: 'Using our tools to research and track potential opportunities.',
    content: `StonkMarket provides several tools to help with your dip research:
    
    1. **Dip Tracker**: Our dashboard shows stocks currently trading below their reference highs, 
    sorted by dip depth and other metrics.
    
    2. **DipSwipe**: A Tinder-style interface to quickly review stocks with AI-generated summaries.
    
    3. **AI Analysis**: GPT-powered insights providing context, ratings, and reasoning for each stock.
    
    4. **Community Voting**: See what other users think about specific dip opportunities.
    
    5. **Benchmark Comparison**: Compare individual stocks against S&P 500 and MSCI World indices.
    
    Remember: Our tools are for research and education only. Always do your own due diligence 
    before making any investment decisions.`,
  },
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
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto space-y-12"
    >
      {/* Hero */}
      <div className="text-center space-y-4">
        <Badge variant="outline" className="mb-2">
          <BookOpen className="mr-1 h-3 w-3" />
          Educational Resources
        </Badge>
        <h1 className="text-4xl font-bold tracking-tight">
          Stock Dip Investing Guide
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Learn the fundamentals of dip buying, how to identify quality opportunities, 
          and the risks every investor should understand.
        </p>
      </div>

      {/* Disclaimer */}
      <Card className="bg-destructive/10 border-destructive/30">
        <CardContent className="pt-6">
          <div className="flex items-start gap-4">
            <AlertTriangle className="h-6 w-6 text-destructive shrink-0 mt-1" />
            <div className="space-y-2">
              <h2 className="text-lg font-semibold text-destructive">Educational Content Only</h2>
              <p className="text-sm text-muted-foreground">
                This content is for educational purposes only and does not constitute financial advice. 
                Investing involves risk, including the potential loss of principal. Past performance 
                does not guarantee future results. Always consult a qualified financial advisor before 
                making investment decisions.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="pt-6 text-center">
            <Clock className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
            <div className="text-2xl font-bold">5 min</div>
            <p className="text-sm text-muted-foreground">Reading time</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6 text-center">
            <BookOpen className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
            <div className="text-2xl font-bold">{ARTICLES.length}</div>
            <p className="text-sm text-muted-foreground">Topics covered</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6 text-center">
            <Shield className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
            <div className="text-2xl font-bold">Free</div>
            <p className="text-sm text-muted-foreground">No registration required</p>
          </CardContent>
        </Card>
      </div>

      {/* Articles */}
      <div className="space-y-8">
        {ARTICLES.map((article, index) => {
          const Icon = article.icon;
          return (
            <Card key={article.id} id={article.id}>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary">
                    {index + 1}
                  </div>
                  <Icon className="h-5 w-5 text-muted-foreground" />
                  {article.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-muted-foreground font-medium">{article.summary}</p>
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {article.content.split('\n\n').map((paragraph, i) => (
                    <p key={i} className="text-sm text-muted-foreground leading-relaxed">
                      {paragraph}
                    </p>
                  ))}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* FAQ Section */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight">Frequently Asked Questions</h2>
        <div className="space-y-4">
          {FAQS.map((faq, index) => (
            <Card key={index}>
              <CardHeader className="pb-2">
                <CardTitle className="text-base font-medium">{faq.question}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{faq.answer}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* CTA */}
      <Card className="bg-muted/50">
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold">Ready to explore?</h3>
              <p className="text-sm text-muted-foreground">
                Try our tools to research stocks currently in a dip.
              </p>
            </div>
            <div className="flex gap-2">
              <Button asChild>
                <Link to="/">
                  View Dashboard
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button variant="outline" asChild>
                <Link to="/swipe">
                  Try DipSwipe
                </Link>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Related Links */}
      <div className="flex flex-wrap gap-2 justify-center">
        <Button variant="ghost" size="sm" asChild>
          <Link to="/about">About Our Methodology</Link>
        </Button>
        <Button variant="ghost" size="sm" asChild>
          <Link to="/privacy">Privacy Policy</Link>
        </Button>
        <Button variant="ghost" size="sm" asChild>
          <Link to="/contact">Contact Us</Link>
        </Button>
      </div>
    </motion.div>
  );
}
