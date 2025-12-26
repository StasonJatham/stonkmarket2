import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  TrendingDown, 
  AlertTriangle,
  BookOpen,
  Target,
  BarChart3,
  Brain,
  Shield,
  ArrowRight,
  Lightbulb,
  Sigma,
  LineChart,
  Activity,
  Calculator,
} from 'lucide-react';
import { useSEO, generateBreadcrumbJsonLd, generateFAQJsonLd } from '@/lib/seo';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

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

// Technical methodology articles
const ARTICLES = [
  {
    id: 'alpha-model',
    title: 'Alpha Model: Ridge/Lasso Ensemble',
    icon: Brain,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    summary: 'Expected return estimation with out-of-sample validation.',
    content: `Our alpha model uses a Ridge/Lasso regularized regression ensemble to predict forward returns over a 2-month horizon.

**Features**: Momentum (21d, 63d, 126d, 252d windows), realized volatility, short-term reversal, and volume trends. All computed with strict no-lookahead guarantees.

**Ensemble Weighting**: Models are combined using inverse-MSE weighting from out-of-sample validation. Each model's weight is proportional to 1/MSE, ensuring better performers contribute more.

**Uncertainty Quantification**: We estimate forecast uncertainty using cross-validation residuals, enabling confidence-aware position sizing.`,
  },
  {
    id: 'dip-score',
    title: 'DipScore: Statistical Dip Detection',
    icon: TrendingDown,
    color: 'text-chart-1',
    bgColor: 'bg-chart-1/10',
    summary: 'Factor-residual z-score for identifying abnormal drawdowns.',
    content: `DipScore measures how unusual a price drop is after controlling for market factors.

**Computation**: DipScore = (r − E[r | factors]) / σ_resid

Where r is the observed return, E[r | factors] is the expected return from factor regression, and σ_resid is the rolling residual volatility.

**Critical Design**: DipScore is **informational only**. It adjusts expected returns (μ_hat) but **never directly generates orders**. All trading decisions flow through the optimizer.

**Bucketing**: Scores are bucketed (≤-2, -2 to -1, -1 to 0, 0 to 1, >1) for regime-conditional analysis.`,
  },
  {
    id: 'risk-model',
    title: 'Risk Model: PCA Factor Covariance',
    icon: Activity,
    color: 'text-purple-500',
    bgColor: 'bg-purple-500/10',
    summary: 'Principal component analysis for portfolio risk estimation.',
    content: `We use a PCA-based factor model to estimate the covariance matrix.

**Structure**: Σ ≈ B Σ_F B^T + D

Where B is the factor loading matrix (n_assets × n_factors), Σ_F is the factor covariance, and D is the diagonal idiosyncratic variance.

**Configuration**: 5 PCA factors by default, capturing major market drivers while filtering noise.

**Marginal Contribution to Risk (MCR)**: For each asset, we compute MCR_i = w_i × (Σw)_i / σ_p, enabling risk-aware allocation decisions.`,
  },
  {
    id: 'optimizer',
    title: 'Portfolio Optimizer: Incremental QP',
    icon: Calculator,
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/10',
    summary: 'Mean-variance optimization with transaction costs.',
    content: `The optimizer solves an incremental mean-variance problem using convex quadratic programming (CVXPY).

**Objective**: max (w+Δw)'μ_hat − λ(w+Δw)'Σ(w+Δw) − TC(Δw)

**Constraints**:
• Long-only: w ≥ 0
• Max position weight: 15%
• Max monthly turnover: 20%
• Transaction cost: €1 fixed per trade
• Minimum trade size: €10

**Output**: Optimal weight changes Δw*, ranked by marginal utility net of transaction costs.`,
  },
  {
    id: 'walk-forward',
    title: 'Validation: Walk-Forward Testing',
    icon: LineChart,
    color: 'text-chart-4',
    bgColor: 'bg-chart-4/10',
    summary: 'Out-of-sample validation with proper time splits.',
    content: `All models are validated using walk-forward methodology to prevent overfitting.

**Split Structure**: 36-month training / 6-month validation / 6-month test windows, rolled forward in 3-month increments.

**Metrics**: Sharpe ratio, maximum drawdown, hit rate (directional accuracy), total turnover.

**Baseline Comparison**: Results compared against equal-weight and 60/40 benchmarks.

**No Lookahead**: Strict separation enforced—no future data leaks into training or feature computation.`,
  },
  {
    id: 'tuning',
    title: 'Hyperparameter Tuning',
    icon: Target,
    color: 'text-primary',
    bgColor: 'bg-primary/10',
    summary: 'Nested walk-forward selection of optimal parameters.',
    content: `Hyperparameters are tuned using nested walk-forward validation.

**Tuned Parameters**:
• Forecast horizon: {1, 2, 3} months
• Ridge alpha: {1, 10, 100}
• Dip coefficient k: {0.001, 0.002, 0.005}
• PCA factors: {3, 5, 8}
• Risk aversion λ: {5, 10, 20}

**Selection Criterion**: Maximize validation Sharpe ratio while penalizing parameter instability.

**Schedule**: Full retuning monthly; dip coefficient can be tuned more frequently if regime shifts detected.`,
  },
];

// Technical quick reference
const QUICK_TIPS = [
  "μ_hat = ensemble-weighted Ridge/Lasso predictions",
  "Σ = PCA factor model with 5 components",
  "DipScore adjusts μ_hat, never generates orders",
  "Max position 15%, max turnover 20%/month",
  "Walk-forward: 36mo train / 6mo val / 6mo test",
];

// FAQs
const FAQS = [
  {
    question: 'Why use Ridge/Lasso instead of more complex models?',
    answer: 'Linear models with regularization are well-suited for noisy financial data. They provide interpretable coefficients, stable out-of-sample performance, and explicit uncertainty quantification. More complex models often overfit without providing better out-of-sample returns.',
  },
  {
    question: "Why doesn't DipScore directly trigger trades?",
    answer: 'DipScore is an informational signal, not a trading rule. It adjusts expected return estimates (μ_hat), which then flow through the optimizer. This ensures all decisions respect constraints (position limits, turnover, costs) and are part of a coherent portfolio optimization.',
  },
  {
    question: 'How do you prevent lookahead bias?',
    answer: 'All features are computed using only past data as of each point in time. Walk-forward validation uses strict temporal separation between train/validation/test sets. The codebase includes explicit no-lookahead verification in tests.',
  },
  {
    question: 'What is the rebalancing frequency?',
    answer: 'The optimizer runs monthly, generating recommendations for the upcoming period. Intra-month, recommendations can be refreshed if significant new data arrives, but the core model is retrained monthly to balance reactivity with stability.',
  },
  {
    question: 'How are transaction costs modeled?',
    answer: "We use a €1 fixed cost per trade (matching TradeRepublic pricing) with a €10 minimum trade size. The optimizer directly incorporates these costs, pruning small trades that don't justify the fixed cost.",
  },
];

// Generate Article schema for SEO
function generateArticleJsonLd(article: typeof ARTICLES[0]) {
  return {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
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
  useSEO({
    title: 'Methodology - Quantitative Portfolio Engine',
    description: 'Technical documentation for our quantitative portfolio optimization engine. Ridge/Lasso alpha models, PCA risk factors, walk-forward validation, and transaction-cost-aware optimization.',
    keywords: 'quantitative finance, portfolio optimization, alpha model, risk model, walk-forward validation, mean-variance optimization',
    canonical: '/learn',
    jsonLd: [
      generateBreadcrumbJsonLd([
        { name: 'Home', url: '/' },
        { name: 'Methodology', url: '/learn' },
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
          <Sigma className="mr-1.5 h-3.5 w-3.5" />
          Quantitative Methodology
        </Badge>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
          Portfolio Engine Documentation
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
          A systematic, mathematically explicit portfolio decision engine. 
          Every signal validated out-of-sample. Every decision from explicit optimization.
        </p>
        
        {/* Quick Stats */}
        <div className="flex flex-wrap justify-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            <span>Walk-forward validated</span>
          </div>
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            <span>{ARTICLES.length} components</span>
          </div>
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            <span>No lookahead bias</span>
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
                <span className="font-semibold text-warning">Research Only:</span>{' '}
                This system is for research and education. Past performance does not guarantee future results. 
                Investing involves risk of loss. Not financial advice.
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
                        if (paragraph.includes('**')) {
                          const parts = paragraph.split(/\*\*(.*?)\*\*/g);
                          return (
                            <p key={i} className="text-sm text-muted-foreground leading-relaxed mb-3 last:mb-0 font-mono">
                              {parts.map((part, j) => 
                                j % 2 === 1 ? <strong key={j} className="text-foreground font-semibold">{part}</strong> : part
                              )}
                            </p>
                          );
                        }
                        return (
                          <p key={i} className="text-sm text-muted-foreground leading-relaxed mb-3 last:mb-0 font-mono">
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
          {/* Quick Reference Card */}
          <motion.div variants={itemVariants}>
            <Card className="sticky top-4">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Lightbulb className="h-4 w-4 text-yellow-500" />
                  Quick Reference
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <ul className="space-y-3">
                  {QUICK_TIPS.map((tip, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-muted-foreground font-mono">
                      <Sigma className="h-3.5 w-3.5 text-primary shrink-0 mt-0.5" />
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
                <CardTitle className="text-base">Components</CardTitle>
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
                        <span className="truncate">{article.title.split(':')[0]}</span>
                      </a>
                    );
                  })}
                </nav>
              </CardContent>
            </Card>
          </motion.div>
        </motion.aside>
      </div>

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
            Technical FAQ
          </h2>
          <p className="text-muted-foreground mt-2">
            Common questions about the methodology
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
                    <p className="text-sm text-muted-foreground leading-relaxed font-mono">
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
                <h3 className="text-xl font-bold mb-2">View Your Portfolio Recommendations</h3>
                <p className="text-muted-foreground max-w-md">
                  Generate optimized allocation recommendations based on this methodology.
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-3">
                <Button asChild size="lg">
                  <Link to="/portfolios">
                    My Portfolios
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <Link to="/">
                    Dashboard
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
            <Link to="/about">About</Link>
          </Button>
          {ENABLE_LEGAL_PAGES && (
            <Button variant="ghost" size="sm" asChild>
              <Link to="/privacy">Privacy Policy</Link>
            </Button>
          )}
          <Button variant="ghost" size="sm" asChild>
            <Link to="/contact">Contact</Link>
          </Button>
        </div>
      </motion.section>
    </div>
  );
}
