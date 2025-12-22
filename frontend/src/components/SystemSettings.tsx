import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  getRuntimeSettings, 
  updateRuntimeSettings, 
  getAppSettings,
  checkOpenAIStatus,
  type RuntimeSettings,
  type AppSettings,
  type BenchmarkConfig
} from '@/services/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
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
} from 'lucide-react';

const AI_MODELS = [
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini (Fast & affordable)' },
  { value: 'gpt-4.1-mini', label: 'GPT-4.1 Mini' },
  { value: 'gpt-4o', label: 'GPT-4o (Best quality)' },
  { value: 'gpt-4.1', label: 'GPT-4.1 (Latest)' },
];

export function SystemSettings() {
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const [runtimeSettings, setRuntimeSettings] = useState<RuntimeSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form state
  const [aiEnabled, setAiEnabled] = useState(false);
  const [aiModel, setAiModel] = useState('gpt-4o-mini');
  const [aiBatchSize, setAiBatchSize] = useState(5);
  const [cleanupDays, setCleanupDays] = useState(30);
  const [strongBuyThreshold, setStrongBuyThreshold] = useState(80);
  const [buyThreshold, setBuyThreshold] = useState(60);
  const [holdThreshold, setHoldThreshold] = useState(40);
  const [benchmarks, setBenchmarks] = useState<BenchmarkConfig[]>([]);
  const [openaiConfigured, setOpenaiConfigured] = useState(false);

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

  const hasChanges = runtimeSettings && (
    aiEnabled !== runtimeSettings.ai_enrichment_enabled ||
    aiModel !== runtimeSettings.ai_model ||
    aiBatchSize !== runtimeSettings.ai_batch_size ||
    cleanupDays !== runtimeSettings.suggestion_cleanup_days ||
    strongBuyThreshold !== runtimeSettings.signal_threshold_strong_buy ||
    buyThreshold !== runtimeSettings.signal_threshold_buy ||
    holdThreshold !== runtimeSettings.signal_threshold_hold ||
    JSON.stringify(benchmarks) !== JSON.stringify(runtimeSettings.benchmarks || [])
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

      {/* AI Settings */}
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
          {/* OpenAI Key Warning */}
          {!openaiConfigured && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-warning/10 text-warning border border-warning/20">
              <KeyRound className="h-4 w-4 shrink-0" />
              <span className="text-sm">
                OpenAI API key not configured. Set <code className="px-1 py-0.5 bg-muted rounded text-xs">OPENAI_API_KEY</code> environment variable or add via API Keys.
              </span>
            </div>
          )}
          
          {/* AI Enable Toggle */}
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

          {/* AI Model Selection */}
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

          {/* Batch Size */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Batch Size</Label>
              <Badge variant="secondary">
                {aiBatchSize === 0 ? 'All stocks' : `${aiBatchSize} per run`}
              </Badge>
            </div>
            <div className="flex items-center gap-4">
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
                ? "All tracked stocks will be analyzed each run"
                : "Number of stocks to analyze per scheduled job run"
              }
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Cleanup Settings */}
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

      {/* Benchmark Management */}
      <BenchmarkManager benchmarks={benchmarks} onChange={setBenchmarks} />

      {/* App Info (Read-only) */}
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
