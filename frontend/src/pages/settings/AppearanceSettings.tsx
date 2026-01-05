import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useTheme } from '@/context/ThemeContext';
import { cn } from '@/lib/utils';
import { Sun, Moon, Monitor, Palette, RotateCcw, TrendingUp, TrendingDown } from 'lucide-react';

export function AppearanceSettings() {
  const {
    theme,
    setTheme,
    colorblindMode,
    setColorblindMode,
    customColors,
    setCustomColors,
    resetColors,
    getActiveColors,
  } = useTheme();

  const activeColors = getActiveColors();

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Appearance</h2>
        <p className="text-sm text-muted-foreground">
          Customize how StonkMarket looks for you
        </p>
      </div>

      {/* Theme Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Theme</CardTitle>
          <CardDescription>Select your preferred color scheme</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <ThemeOption
              value="light"
              current={theme}
              onSelect={setTheme}
              icon={Sun}
              label="Light"
            />
            <ThemeOption
              value="dark"
              current={theme}
              onSelect={setTheme}
              icon={Moon}
              label="Dark"
            />
            <ThemeOption
              value="system"
              current={theme}
              onSelect={setTheme}
              icon={Monitor}
              label="System"
            />
          </div>
        </CardContent>
      </Card>

      {/* Color Blind Mode */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Palette className="h-5 w-5" />
            Accessibility
          </CardTitle>
          <CardDescription>
            Adjust colors for better visibility
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Color Blind Mode</Label>
              <p className="text-sm text-muted-foreground">
                Use blue/orange instead of green/red for better contrast
              </p>
            </div>
            <Switch
              checked={colorblindMode}
              onCheckedChange={setColorblindMode}
            />
          </div>

          {colorblindMode && (
            <div className="p-3 bg-muted/50 rounded-lg">
              <p className="text-sm text-muted-foreground">
                Color blind mode is active. Custom colors are disabled.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Custom Chart Colors */}
      <Card>
        <CardHeader>
          <CardTitle>Chart Colors</CardTitle>
          <CardDescription>
            Customize gain and loss indicator colors
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label>Gain Color (Up)</Label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={customColors.up}
                  onChange={(e) => setCustomColors({ ...customColors, up: e.target.value })}
                  disabled={colorblindMode}
                  className="h-10 w-14 rounded border border-input cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <div
                  className="flex-1 h-10 rounded-md flex items-center justify-center gap-2"
                  style={{ backgroundColor: `${activeColors.up}20`, color: activeColors.up }}
                >
                  <TrendingUp className="h-4 w-4" />
                  <span className="font-medium">+5.25%</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Loss Color (Down)</Label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={customColors.down}
                  onChange={(e) => setCustomColors({ ...customColors, down: e.target.value })}
                  disabled={colorblindMode}
                  className="h-10 w-14 rounded border border-input cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <div
                  className="flex-1 h-10 rounded-md flex items-center justify-center gap-2"
                  style={{ backgroundColor: `${activeColors.down}20`, color: activeColors.down }}
                >
                  <TrendingDown className="h-4 w-4" />
                  <span className="font-medium">-3.12%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Live Preview */}
          <div className="p-4 bg-card border rounded-lg">
            <div className="text-sm font-medium mb-3">Live Preview</div>
            <div className="flex gap-4">
              <Badge
                variant="outline"
                style={{ 
                  backgroundColor: `${activeColors.up}15`,
                  borderColor: `${activeColors.up}50`,
                  color: activeColors.up 
                }}
              >
                AAPL +2.5%
              </Badge>
              <Badge
                variant="outline"
                style={{ 
                  backgroundColor: `${activeColors.down}15`,
                  borderColor: `${activeColors.down}50`,
                  color: activeColors.down 
                }}
              >
                MSFT -1.2%
              </Badge>
              <Badge
                variant="outline"
                style={{ 
                  backgroundColor: `${activeColors.up}15`,
                  borderColor: `${activeColors.up}50`,
                  color: activeColors.up 
                }}
              >
                NVDA +5.8%
              </Badge>
            </div>
          </div>

          <Button
            variant="outline"
            onClick={resetColors}
            disabled={colorblindMode}
            className="gap-2"
          >
            <RotateCcw className="h-4 w-4" />
            Reset to Defaults
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

interface ThemeOptionProps {
  value: 'light' | 'dark' | 'system';
  current: string;
  onSelect: (value: 'light' | 'dark' | 'system') => void;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
}

function ThemeOption({ value, current, onSelect, icon: Icon, label }: ThemeOptionProps) {
  const isActive = current === value;

  return (
    <button
      onClick={() => onSelect(value)}
      className={cn(
        'flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-colors',
        isActive
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/50 hover:bg-muted/50'
      )}
    >
      <Icon className={cn('h-6 w-6', isActive && 'text-primary')} />
      <span className={cn('text-sm font-medium', isActive && 'text-primary')}>
        {label}
      </span>
    </button>
  );
}

export default AppearanceSettings;
