import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  getRuntimeSettings, 
  updateRuntimeSettings, 
  getAppSettings,
  checkOpenAIStatus,
  type RuntimeSettings,
  type AppSettings,
  type BenchmarkConfig,
  type SectorETFConfig
} from '@/services/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import { BenchmarkManager } from '@/components/BenchmarkManager';
import { SectorETFManager } from '@/components/SectorETFManager';
import { ApiKeyManager } from '@/components/ApiKeyManager';
import { 
  Sparkles, 
  Save, 
  Loader2, 
  Info, 
  Bot, 
  Trash2,
  AlertCircle,
  CheckCircle,
  KeyRound,
  DollarSign,
  Percent,
  BarChart2,
} from 'lucide-react';

const AI_MODELS = [
  { value: 'gpt-5-mini', label: 'GPT-5 Mini (Fast & affordable)' },
  { value: 'gpt-5', label: 'GPT-5 (Best quality)' },
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini (Legacy)' },
  { value: 'gpt-4o', label: 'GPT-4o (Legacy)' },
];

interface SystemSettingsProps {
  defaultTab?: 'ai' | 'trading' | 'benchmarks' | 'maintenance' | 'integrations' | 'about';
  showTabs?: boolean;
}

type SectionKey = NonNullable<SystemSettingsProps['defaultTab']>;

export function SystemSettings({ defaultTab = 'ai', showTabs = true }: SystemSettingsProps) {
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const [runtimeSettings, setRuntimeSettings] = useState<RuntimeSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form state
  const [aiEnabled, setAiEnabled] = useState(false);
  const [aiModel, setAiModel] = useState('gpt-5-mini');
  const [aiBatchSize, setAiBatchSize] = useState(5);
  const [cleanupDays, setCleanupDays] = useState(30);
  const [strongBuyThreshold, setStrongBuyThreshold] = useState(80);
  const [buyThreshold, setBuyThreshold] = useState(60);
  const [holdThreshold, setHoldThreshold] = useState(40);
  const [benchmarks, setBenchmarks] = useState<BenchmarkConfig[]>([]);
  const [sectorEtfs, setSectorEtfs] = useState<SectorETFConfig[]>([]);
  const [openaiConfigured, setOpenaiConfigured] = useState(false);

  // Trading/Backtest form state
  const [initialCapital, setInitialCapital] = useState(50000);
  const [flatCostPerTrade, setFlatCostPerTrade] = useState(1.0);
  const [slippageBps, setSlippageBps] = useState(5.0);
  const [stopLossPct, setStopLossPct] = useState(15.0);
  const [takeProfitPct, setTakeProfitPct] = useState(30.0);
  const [maxHoldingDays, setMaxHoldingDays] = useState(120);
  const [minTradesRequired, setMinTradesRequired] = useState(30);
  const [walkForwardFolds, setWalkForwardFolds] = useState(5);
  const [trainRatio, setTrainRatio] = useState(0.70);

  useEffect(() => {
    loadSettings();
  }, []);

  async function loadSettings() {
    setIsLoading(true);
    setError(null);
    try {
      const [app, runtime, openaiStatus] = await Promise.all([
        getAppSettings(),
        getRuntimeSettings(),
        checkOpenAIStatus(),
      ]);
      setAppSettings(app);
      setRuntimeSettings(runtime);
      setOpenaiConfigured(openaiStatus.configured);
      
      // Initialize form state
      setAiEnabled(runtime.ai_enrichment_enabled);
      setAiModel(runtime.ai_model);
      setAiBatchSize(runtime.ai_batch_size);
      setCleanupDays(runtime.suggestion_cleanup_days);
      setStrongBuyThreshold(runtime.signal_threshold_strong_buy);
      setBuyThreshold(runtime.signal_threshold_buy);
      setHoldThreshold(runtime.signal_threshold_hold);
      setBenchmarks(runtime.benchmarks || []);
      setSectorEtfs(runtime.sector_etfs || []);
      
      // Initialize trading/backtest settings
      setInitialCapital(runtime.trading_initial_capital);
      setFlatCostPerTrade(runtime.trading_flat_cost_per_trade);
      setSlippageBps(runtime.trading_slippage_bps);
      setStopLossPct(runtime.trading_stop_loss_pct);
      setTakeProfitPct(runtime.trading_take_profit_pct);
      setMaxHoldingDays(runtime.trading_max_holding_days);
      setMinTradesRequired(runtime.trading_min_trades_required);
      setWalkForwardFolds(runtime.trading_walk_forward_folds);
      setTrainRatio(runtime.trading_train_ratio);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load settings');
    } finally {
      setIsLoading(false);
    }
  }

  async function handleSave() {
    setIsSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const updated = await updateRuntimeSettings({
        ai_enrichment_enabled: aiEnabled,
        ai_model: aiModel,
        ai_batch_size: aiBatchSize,
        suggestion_cleanup_days: cleanupDays,
        signal_threshold_strong_buy: strongBuyThreshold,
        signal_threshold_buy: buyThreshold,
        signal_threshold_hold: holdThreshold,
        benchmarks: benchmarks,
        // Trading/Backtest settings
        trading_initial_capital: initialCapital,
        trading_flat_cost_per_trade: flatCostPerTrade,
        trading_slippage_bps: slippageBps,
        trading_stop_loss_pct: stopLossPct,
        trading_take_profit_pct: takeProfitPct,
        trading_max_holding_days: maxHoldingDays,
        trading_min_trades_required: minTradesRequired,
        trading_walk_forward_folds: walkForwardFolds,
        trading_train_ratio: trainRatio,
      });
      setRuntimeSettings(updated);
      setSuccess('Settings saved successfully!');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  }

  // Save benchmarks immediately when edited in BenchmarkManager
  async function saveBenchmarks(newBenchmarks: BenchmarkConfig[]) {
    setIsSaving(true);
    setError(null);
    try {
      const updated = await updateRuntimeSettings({ benchmarks: newBenchmarks });
      setRuntimeSettings(updated);
      setSuccess('Benchmark saved!');
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save benchmark');
    } finally {
      setIsSaving(false);
    }
  }

  // Save sector ETFs immediately when edited in SectorETFManager
  async function saveSectorEtfs(newSectorEtfs: SectorETFConfig[]) {
    setIsSaving(true);
    setError(null);
    try {
      const updated = await updateRuntimeSettings({ sector_etfs: newSectorEtfs });
      setRuntimeSettings(updated);
      setSuccess('Sector ETF saved!');
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save sector ETF');
    } finally {
      setIsSaving(false);
    }
  }

  const hasChanges = runtimeSettings && (
    aiEnabled !== runtimeSettings.ai_enrichment_enabled ||
    aiModel !== runtimeSettings.ai_model ||
    aiBatchSize !== runtimeSettings.ai_batch_size ||
    cleanupDays !== runtimeSettings.suggestion_cleanup_days ||
    strongBuyThreshold !== runtimeSettings.signal_threshold_strong_buy ||
    buyThreshold !== runtimeSettings.signal_threshold_buy ||
    holdThreshold !== runtimeSettings.signal_threshold_hold ||
    JSON.stringify(benchmarks) !== JSON.stringify(runtimeSettings.benchmarks || []) ||
    // Trading/Backtest settings
    initialCapital !== runtimeSettings.trading_initial_capital ||
    flatCostPerTrade !== runtimeSettings.trading_flat_cost_per_trade ||
    slippageBps !== runtimeSettings.trading_slippage_bps ||
    stopLossPct !== runtimeSettings.trading_stop_loss_pct ||
    takeProfitPct !== runtimeSettings.trading_take_profit_pct ||
    maxHoldingDays !== runtimeSettings.trading_max_holding_days ||
    minTradesRequired !== runtimeSettings.trading_min_trades_required ||
    walkForwardFolds !== runtimeSettings.trading_walk_forward_folds ||
    trainRatio !== runtimeSettings.trading_train_ratio
  );

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-48 rounded-xl" />
        <Skeleton className="h-48 rounded-xl" />
        <Skeleton className="h-32 rounded-xl" />
      </div>
    );
  }

  const aiContent = (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            AI Enrichment
          </CardTitle>
          <CardDescription>
            Configure AI-powered stock analysis and enrichment features.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {!openaiConfigured && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-warning/10 text-warning border border-warning/20">
              <KeyRound className="h-4 w-4 shrink-0" />
              <span className="text-sm">
                OpenAI API key not configured. Set <code className="px-1 py-0.5 bg-muted rounded text-xs">OPENAI_API_KEY</code> environment variable or add via API Keys.
              </span>
            </div>
          )}

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label className="text-base">Enable AI Enrichment</Label>
              <p className="text-sm text-muted-foreground">
                Generate AI-powered stock insights and bios
              </p>
            </div>
            <Switch
              checked={aiEnabled}
              onCheckedChange={setAiEnabled}
              disabled={!openaiConfigured}
            />
          </div>

          <Separator />

          <div className="space-y-3">
            <Label className="flex items-center gap-2">
              <Bot className="h-4 w-4" />
              AI Model
            </Label>
            <Select value={aiModel} onValueChange={setAiModel} disabled={!aiEnabled}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {AI_MODELS.map((model) => (
                  <SelectItem key={model.value} value={model.value}>
                    {model.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Higher quality models cost more but provide better analysis
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            Batch Processing
          </CardTitle>
          <CardDescription>
            Control AI batch throughput and workload size.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <Label>Batch Size</Label>
            <Badge variant="secondary">
              {aiBatchSize === 0 ? 'All stocks' : `${aiBatchSize} per run`}
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="batch-all"
              checked={aiBatchSize === 0}
              onChange={(e) => setAiBatchSize(e.target.checked ? 0 : 5)}
              disabled={!aiEnabled}
              className="h-4 w-4 rounded border-border"
            />
            <Label htmlFor="batch-all" className="text-sm font-normal cursor-pointer">
              Process all stocks
            </Label>
          </div>

          {aiBatchSize > 0 && (
            <Slider
              value={[aiBatchSize]}
              onValueChange={([v]) => setAiBatchSize(v)}
              min={1}
              max={50}
              step={1}
              disabled={!aiEnabled}
            />
          )}

          <p className="text-xs text-muted-foreground">
            {aiBatchSize === 0
              ? 'All tracked stocks will be analyzed each run'
              : 'Number of stocks to analyze per scheduled job run'}
          </p>
        </CardContent>
      </Card>
    </div>
  );

  const tradingContent = (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart2 className="h-5 w-5 text-primary" />
            Signal Thresholds
          </CardTitle>
          <CardDescription>
            Adjust score thresholds for recommendations.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="threshold-strong-buy">Strong Buy</Label>
            <Input
              id="threshold-strong-buy"
              type="number"
              value={strongBuyThreshold}
              onChange={(e) => setStrongBuyThreshold(parseInt(e.target.value) || 80)}
              min={1}
              max={100}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="threshold-buy">Buy</Label>
            <Input
              id="threshold-buy"
              type="number"
              value={buyThreshold}
              onChange={(e) => setBuyThreshold(parseInt(e.target.value) || 60)}
              min={1}
              max={100}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="threshold-hold">Hold</Label>
            <Input
              id="threshold-hold"
              type="number"
              value={holdThreshold}
              onChange={(e) => setHoldThreshold(parseInt(e.target.value) || 40)}
              min={1}
              max={100}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="h-5 w-5 text-primary" />
            Capital & Costs
          </CardTitle>
          <CardDescription>
            Tune capital assumptions and trading costs.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="initial-capital">Initial Capital (€)</Label>
              <Input
                id="initial-capital"
                type="number"
                value={initialCapital}
                onChange={(e) => setInitialCapital(parseFloat(e.target.value) || 50000)}
                min={1000}
                max={10000000}
                step={1000}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="flat-cost">Flat Cost (€/trade)</Label>
              <Input
                id="flat-cost"
                type="number"
                value={flatCostPerTrade}
                onChange={(e) => setFlatCostPerTrade(parseFloat(e.target.value) || 1)}
                min={0}
                max={100}
                step={0.5}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="slippage">Slippage (bps)</Label>
              <Input
                id="slippage"
                type="number"
                value={slippageBps}
                onChange={(e) => setSlippageBps(parseFloat(e.target.value) || 5)}
                min={0}
                max={100}
                step={1}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Flat cost per trade in euros. Slippage is additional per-side cost in basis points (1 bp = 0.01%).
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Percent className="h-5 w-5 text-primary" />
            Risk Management
          </CardTitle>
          <CardDescription>
            Define exit and risk limits for backtests.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="stop-loss">Stop Loss (%)</Label>
              <Input
                id="stop-loss"
                type="number"
                value={stopLossPct}
                onChange={(e) => setStopLossPct(parseFloat(e.target.value) || 15)}
                min={1}
                max={50}
                step={1}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="take-profit">Take Profit (%)</Label>
              <Input
                id="take-profit"
                type="number"
                value={takeProfitPct}
                onChange={(e) => setTakeProfitPct(parseFloat(e.target.value) || 30)}
                min={5}
                max={100}
                step={1}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="max-holding">Max Holding (days)</Label>
              <Input
                id="max-holding"
                type="number"
                value={maxHoldingDays}
                onChange={(e) => setMaxHoldingDays(parseInt(e.target.value) || 120)}
                min={5}
                max={365}
                step={5}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Exit rules: Stop loss when price drops by %, take profit when it rises by %, or force exit after max holding days.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart2 className="h-5 w-5 text-primary" />
            Validation & Walk-Forward
          </CardTitle>
          <CardDescription>
            Statistical requirements for backtest validity.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="min-trades">Min Trades Required</Label>
              <Input
                id="min-trades"
                type="number"
                value={minTradesRequired}
                onChange={(e) => setMinTradesRequired(parseInt(e.target.value) || 30)}
                min={10}
                max={200}
                step={5}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="walk-forward">Walk-Forward Folds</Label>
              <Input
                id="walk-forward"
                type="number"
                value={walkForwardFolds}
                onChange={(e) => setWalkForwardFolds(parseInt(e.target.value) || 5)}
                min={2}
                max={10}
                step={1}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="train-ratio">Train Ratio</Label>
              <Input
                id="train-ratio"
                type="number"
                value={trainRatio}
                onChange={(e) => setTrainRatio(parseFloat(e.target.value) || 0.70)}
                min={0.50}
                max={0.90}
                step={0.05}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Minimum trades for statistical significance. Walk-forward validation uses rolling train/test windows with the specified train ratio.
          </p>
        </CardContent>
      </Card>
    </div>
  );

  const benchmarksContent = (
    <div className="space-y-6">
      <BenchmarkManager
        benchmarks={benchmarks}
        onChange={setBenchmarks}
        onSave={saveBenchmarks}
      />
      <SectorETFManager
        sectorEtfs={sectorEtfs}
        onChange={setSectorEtfs}
        onSave={saveSectorEtfs}
      />
    </div>
  );

  const maintenanceContent = (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Trash2 className="h-5 w-5 text-primary" />
          Cleanup Settings
        </CardTitle>
        <CardDescription>
          Configure automatic data cleanup and retention.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-3">
          <Label>Suggestion Cleanup (days)</Label>
          <Input
            type="number"
            value={cleanupDays}
            onChange={(e) => setCleanupDays(parseInt(e.target.value) || 30)}
            min={7}
            max={365}
            className="max-w-32"
          />
          <p className="text-xs text-muted-foreground">
            Remove rejected and pending suggestions older than this many days
          </p>
        </div>
      </CardContent>
    </Card>
  );

  const integrationsContent = (
    <ApiKeyManager onError={setError} onSuccess={setSuccess} />
  );

  const aboutContent = (
    <>
      {appSettings && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="h-5 w-5 text-muted-foreground" />
              Application Info
            </CardTitle>
            <CardDescription>
              Read-only application configuration.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 text-sm">
              <div className="flex items-center justify-between py-2 border-b">
                <span className="text-muted-foreground">App Name</span>
                <span className="font-medium">{appSettings.app_name}</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b">
                <span className="text-muted-foreground">Version</span>
                <Badge variant="outline">{appSettings.app_version}</Badge>
              </div>
              <div className="flex items-center justify-between py-2 border-b">
                <span className="text-muted-foreground">Environment</span>
                <Badge variant={appSettings.environment === 'production' ? 'default' : 'secondary'}>
                  {appSettings.environment}
                </Badge>
              </div>
              <div className="flex items-center justify-between py-2 border-b">
                <span className="text-muted-foreground">Scheduler</span>
                <Badge variant={appSettings.scheduler_enabled ? 'default' : 'secondary'}>
                  {appSettings.scheduler_enabled ? 'Enabled' : 'Disabled'}
                </Badge>
              </div>
              <div className="flex items-center justify-between py-2">
                <span className="text-muted-foreground">Timezone</span>
                <span className="font-medium">{appSettings.scheduler_timezone}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );

  const sectionContent: Record<SectionKey, React.JSX.Element> = {
    ai: aiContent,
    trading: tradingContent,
    benchmarks: benchmarksContent,
    maintenance: maintenanceContent,
    integrations: integrationsContent,
    about: aboutContent,
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Alerts */}
      {error && (
        <div className="flex items-center gap-2 p-4 rounded-xl bg-danger/10 text-danger border border-danger/20">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
      
      {success && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="flex items-center gap-2 p-4 rounded-xl bg-success/10 text-success border border-success/20"
        >
          <CheckCircle className="h-4 w-4 shrink-0" />
          <span>{success}</span>
        </motion.div>
      )}

      {showTabs ? (
        <Tabs defaultValue={defaultTab} className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
            <TabsList className="flex flex-wrap gap-2 lg:flex-col lg:items-stretch">
              <TabsTrigger value="ai" className="justify-start">AI</TabsTrigger>
              <TabsTrigger value="trading" className="justify-start">Trading</TabsTrigger>
              <TabsTrigger value="benchmarks" className="justify-start">Benchmarks</TabsTrigger>
              <TabsTrigger value="maintenance" className="justify-start">Maintenance</TabsTrigger>
              <TabsTrigger value="integrations" className="justify-start">Integrations</TabsTrigger>
              <TabsTrigger value="about" className="justify-start">About</TabsTrigger>
            </TabsList>

            <div className="space-y-6">
              <TabsContent value="ai" className="space-y-6">
                {sectionContent.ai}
              </TabsContent>
              <TabsContent value="trading" className="space-y-6">
                {sectionContent.trading}
              </TabsContent>
              <TabsContent value="benchmarks" className="space-y-6">
                {sectionContent.benchmarks}
              </TabsContent>
              <TabsContent value="maintenance" className="space-y-6">
                {sectionContent.maintenance}
              </TabsContent>
              <TabsContent value="integrations" className="space-y-6">
                {sectionContent.integrations}
              </TabsContent>
              <TabsContent value="about" className="space-y-6">
                {sectionContent.about}
              </TabsContent>
            </div>
          </div>
        </Tabs>
      ) : (
        <div className="space-y-6">
          {sectionContent[defaultTab]}
        </div>
      )}

      {/* Save Button */}
      <div className="flex justify-end">
        <Button
          onClick={handleSave}
          disabled={isSaving || !hasChanges}
          className="min-w-32"
        >
          {isSaving ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Save className="h-4 w-4 mr-2" />
          )}
          Save Changes
        </Button>
      </div>
    </motion.div>
  );
}
