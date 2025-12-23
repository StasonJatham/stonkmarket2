import { motion } from 'framer-motion';
import { 
  TrendingDown, 
  BarChart3, 
  Brain, 
  Users, 
  Target, 
  AlertTriangle,
  Database,
  RefreshCw,
  ChartLine,
  Scale
} from 'lucide-react';
import { useSEO, generateBreadcrumbJsonLd, generateFAQJsonLd } from '@/lib/seo';

// FAQs for structured data
const FAQS = [
  {
    question: 'What is a stock dip?',
    answer: 'A stock dip refers to when a stock price falls significantly below its recent high (typically 52-week high). StonkMarket tracks stocks that have dropped 10% or more from their peak, identifying potential buying opportunities.',
  },
  {
    question: 'How does StonkMarket calculate dip scores?',
    answer: 'Our dip score combines multiple factors: the percentage drop from 52-week high, recent recovery momentum, trading volume patterns, and AI-generated sentiment analysis. Higher scores indicate stocks with stronger recovery potential.',
  },
  {
    question: 'Is StonkMarket financial advice?',
    answer: 'No. StonkMarket is an educational and analytical tool only. We do not provide financial advice, investment recommendations, or personalized guidance. Always consult a licensed financial advisor before making investment decisions.',
  },
  {
    question: 'Where does the stock data come from?',
    answer: 'Stock price data is sourced from reputable financial data providers via Yahoo Finance APIs. Data is updated regularly throughout trading hours. Historical data may have slight delays.',
  },
  {
    question: 'How does the AI analysis work?',
    answer: 'Our AI analysis uses OpenAI GPT models to generate summaries, sentiment ratings, and "dating app style" bios for stocks. The AI considers fundamental factors, market conditions, and recent news. AI outputs are informational only and not investment advice.',
  },
];

export function AboutPage() {
  // SEO with FAQ structured data
  useSEO({
    title: 'About StonkMarket - Methodology & Data Sources',
    description: 'Learn how StonkMarket tracks stock dips, our methodology for identifying recovery potential, data sources, and the AI analysis behind our insights.',
    keywords: 'stock dip methodology, investment analysis, data sources, AI stock analysis, dip buying strategy',
    canonical: '/about',
    jsonLd: [
      generateBreadcrumbJsonLd([
        { name: 'Home', url: '/' },
        { name: 'About', url: '/about' },
      ]),
      generateFAQJsonLd(FAQS),
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
        <h1 className="text-4xl font-bold tracking-tight">About StonkMarket</h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          A free, open-source tool for tracking stock market dips and identifying 
          potential recovery opportunities. Built for educational purposes.
        </p>
      </div>

      {/* Risk Disclaimer - Prominent for E-E-A-T */}
      <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-6 risk-disclaimer">
        <div className="flex items-start gap-4">
          <AlertTriangle className="h-6 w-6 text-destructive shrink-0 mt-1" />
          <div className="space-y-2">
            <h2 className="text-lg font-semibold text-destructive">Important Disclaimer</h2>
            <p className="text-sm disclaimer-text">
              <strong>This is not financial advice.</strong> StonkMarket is an educational and 
              informational tool only. We do not provide investment advice, recommendations, 
              or personalized financial guidance. Past performance does not guarantee future 
              results. Stock investments carry significant risk, including the potential loss 
              of principal. Always consult a qualified financial advisor before making any 
              investment decisions.
            </p>
          </div>
        </div>
      </div>

      {/* What We Do */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Target className="h-6 w-6" />
          What We Track
        </h2>
        <div className="grid sm:grid-cols-2 gap-6">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <TrendingDown className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Stock Dips</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              We monitor stocks that have fallen significantly from their 52-week highs. 
              A "dip" typically means a drop of 10% or more, which may indicate a buying 
              opportunity for value-oriented investors.
            </p>
          </div>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <ChartLine className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Recovery Potential</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              We track recovery momentum using technical indicators like price action, 
              volume patterns, and moving averages to identify stocks showing signs of 
              bouncing back.
            </p>
          </div>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Benchmark Comparison</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Compare individual stocks or your watchlist against major benchmarks like 
              the S&P 500 and MSCI World Index to understand relative performance.
            </p>
          </div>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Users className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Community Sentiment</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Our DipSwipe feature allows users to vote on whether a stock dip looks 
              like a "buy" or "sell" opportunity, creating crowd-sourced sentiment data.
            </p>
          </div>
        </div>
      </section>

      {/* Methodology */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Scale className="h-6 w-6" />
          Our Methodology
        </h2>
        <div className="space-y-4 text-sm text-muted-foreground">
          <div className="bg-muted/50 p-4 rounded-lg space-y-3">
            <h3 className="font-semibold text-foreground">Dip Score Calculation</h3>
            <p>
              Our proprietary dip score (0-100) combines multiple factors:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Dip Depth (40%)</strong> - How far the stock has fallen from its 52-week high</li>
              <li><strong>Recovery Momentum (30%)</strong> - Recent price action showing upward movement</li>
              <li><strong>Volume Analysis (15%)</strong> - Trading volume patterns indicating accumulation</li>
              <li><strong>AI Sentiment (15%)</strong> - AI-generated analysis of news and fundamentals</li>
            </ul>
          </div>
          <p>
            Stocks are ranked by their composite score, with higher scores indicating 
            what our algorithm considers stronger recovery potential. This is a quantitative 
            assessment only and should not be interpreted as investment advice.
          </p>
        </div>
      </section>

      {/* AI Analysis */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Brain className="h-6 w-6" />
          AI-Powered Insights
        </h2>
        <div className="space-y-4 text-sm text-muted-foreground">
          <p>
            We use OpenAI's GPT models to generate supplementary analysis for stocks:
          </p>
          <div className="grid sm:grid-cols-3 gap-4">
            <div className="bg-muted/50 p-4 rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">Stock Summaries</h4>
              <p>AI-generated summaries of company fundamentals, recent news, and market position.</p>
            </div>
            <div className="bg-muted/50 p-4 rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">Sentiment Rating</h4>
              <p>AI assessment of bullish, bearish, or neutral outlook based on available data.</p>
            </div>
            <div className="bg-muted/50 p-4 rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">DipSwipe Bios</h4>
              <p>Fun, "dating app style" stock descriptions for our swipe-to-vote feature.</p>
            </div>
          </div>
          <p className="text-xs">
            <strong>Note:</strong> AI-generated content is informational only. It may contain 
            inaccuracies and should not be relied upon for investment decisions. Always verify 
            information with authoritative financial sources.
          </p>
        </div>
      </section>

      {/* Data Sources */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Database className="h-6 w-6" />
          Data Sources
        </h2>
        <div className="space-y-4 text-sm text-muted-foreground">
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b">
                  <th className="pb-2 font-semibold text-foreground">Data Type</th>
                  <th className="pb-2 font-semibold text-foreground">Source</th>
                  <th className="pb-2 font-semibold text-foreground">Update Frequency</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                <tr>
                  <td className="py-2">Stock Prices</td>
                  <td className="py-2">Yahoo Finance API</td>
                  <td className="py-2">Every 15-30 minutes during market hours</td>
                </tr>
                <tr>
                  <td className="py-2">Historical Data</td>
                  <td className="py-2">Yahoo Finance API</td>
                  <td className="py-2">Daily (end of day)</td>
                </tr>
                <tr>
                  <td className="py-2">Company Information</td>
                  <td className="py-2">Yahoo Finance API</td>
                  <td className="py-2">Weekly refresh</td>
                </tr>
                <tr>
                  <td className="py-2">AI Analysis</td>
                  <td className="py-2">OpenAI GPT-4</td>
                  <td className="py-2">On-demand / batch processing</td>
                </tr>
                <tr>
                  <td className="py-2">Community Votes</td>
                  <td className="py-2">User-generated</td>
                  <td className="py-2">Real-time</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs">
            Data accuracy is not guaranteed. Prices may be delayed. Always verify critical 
            information with your broker or authoritative financial sources before trading.
          </p>
        </div>
      </section>

      {/* Update Schedule */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <RefreshCw className="h-6 w-6" />
          Update Schedule
        </h2>
        <div className="space-y-4 text-sm text-muted-foreground">
          <ul className="space-y-2">
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full"></span>
              <strong>Dip Rankings:</strong> Refreshed every 30 minutes during US market hours
            </li>
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full"></span>
              <strong>Stock Charts:</strong> Updated every 15 minutes with intraday data
            </li>
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-yellow-500 rounded-full"></span>
              <strong>AI Analysis:</strong> Regenerated weekly or on-demand
            </li>
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
              <strong>Benchmarks:</strong> S&P 500 and MSCI World updated daily
            </li>
          </ul>
        </div>
      </section>

      {/* FAQs */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold">Frequently Asked Questions</h2>
        <div className="space-y-4">
          {FAQS.map((faq, index) => (
            <div key={index} className="border rounded-lg p-4 space-y-2">
              <h3 className="font-semibold">{faq.question}</h3>
              <p className="text-sm text-muted-foreground">{faq.answer}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Final Disclaimer */}
      <section className="border-t pt-8 text-center text-sm text-muted-foreground space-y-4">
        <p>
          StonkMarket is a personal, non-commercial project created for educational purposes. 
          It is not affiliated with any financial institution, broker, or investment firm.
        </p>
        <p className="font-semibold text-foreground">
          Invest responsibly. Never invest more than you can afford to lose.
        </p>
      </section>
    </motion.div>
  );
}
