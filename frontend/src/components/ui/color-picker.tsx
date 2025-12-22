import * as React from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { cn } from '@/lib/utils';
import { Paintbrush, Check, RotateCcw, TrendingUp, TrendingDown } from 'lucide-react';

interface ColorPickerProps {
  value: string;
  onChange: (color: string) => void;
  label?: string;
  presets?: string[];
  className?: string;
  disabled?: boolean;
}

const DEFAULT_PRESETS = [
  '#22c55e', // Green (default up)
  '#ef4444', // Red (default down)
  '#3b82f6', // Blue (colorblind up)
  '#f97316', // Orange (colorblind down)
  '#8b5cf6', // Purple
  '#ec4899', // Pink
  '#14b8a6', // Teal
  '#eab308', // Yellow
  '#06b6d4', // Cyan
  '#64748b', // Slate
];

export function ColorPicker({
  value,
  onChange,
  label,
  presets = DEFAULT_PRESETS,
  className,
  disabled = false,
}: ColorPickerProps) {
  const [inputValue, setInputValue] = React.useState(value);

  React.useEffect(() => {
    setInputValue(value);
  }, [value]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);
    // Only update if it's a valid hex color
    if (/^#[0-9A-Fa-f]{6}$/.test(newValue)) {
      onChange(newValue);
    }
  };

  const handleInputBlur = () => {
    // Reset to current value if invalid
    if (!/^#[0-9A-Fa-f]{6}$/.test(inputValue)) {
      setInputValue(value);
    }
  };

  const handlePresetClick = (color: string) => {
    setInputValue(color);
    onChange(color);
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn(
            'h-9 gap-2 px-3 font-normal',
            disabled && 'opacity-50 cursor-not-allowed',
            className
          )}
          disabled={disabled}
        >
          <div
            className="h-4 w-4 rounded-sm border border-border/50 shadow-sm"
            style={{ backgroundColor: value }}
          />
          {label && <span className="text-sm">{label}</span>}
          <Paintbrush className="h-3.5 w-3.5 ml-auto text-muted-foreground" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-64 p-3" align="start">
        <div className="space-y-3">
          {/* Color preview and hex input */}
          <div className="flex items-center gap-3">
            <div
              className="h-10 w-10 rounded-md border border-border shadow-sm shrink-0"
              style={{ backgroundColor: value }}
            />
            <div className="flex-1 space-y-1">
              {label && (
                <Label className="text-xs text-muted-foreground">{label}</Label>
              )}
              <Input
                type="text"
                value={inputValue}
                onChange={handleInputChange}
                onBlur={handleInputBlur}
                placeholder="#000000"
                className="h-8 font-mono text-sm"
                maxLength={7}
              />
            </div>
          </div>

          {/* Native color picker */}
          <div className="relative">
            <Input
              type="color"
              value={value}
              onChange={(e) => {
                const newColor = e.target.value;
                setInputValue(newColor);
                onChange(newColor);
              }}
              className="h-8 w-full cursor-pointer p-0 border-0"
            />
          </div>

          {/* Preset colors */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Presets</Label>
            <div className="grid grid-cols-5 gap-1.5">
              {presets.map((color) => (
                <button
                  key={color}
                  type="button"
                  className={cn(
                    'h-7 w-7 rounded-md border-2 transition-all hover:scale-110 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
                    value === color
                      ? 'border-foreground'
                      : 'border-transparent hover:border-border'
                  )}
                  style={{ backgroundColor: color }}
                  onClick={() => handlePresetClick(color)}
                  title={color}
                >
                  {value === color && (
                    <Check className="h-3.5 w-3.5 mx-auto text-white drop-shadow-md" />
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

interface ColorPickerInlineProps {
  upColor: string;
  downColor: string;
  onUpChange: (color: string) => void;
  onDownChange: (color: string) => void;
  onReset?: () => void;
  className?: string;
}

// Minimal color picker trigger - just a colored dot
function ColorDot({
  value,
  onChange,
  icon: Icon,
  presets,
  title,
}: {
  value: string;
  onChange: (color: string) => void;
  icon: React.ElementType;
  presets: string[];
  title: string;
}) {
  const handlePresetClick = (color: string) => {
    onChange(color);
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          className="flex items-center gap-1 hover:opacity-80 transition-opacity"
          title={title}
        >
          <Icon className="h-3 w-3" style={{ color: value }} />
          <div
            className="h-3 w-3 rounded-full border border-border/50 shadow-sm"
            style={{ backgroundColor: value }}
          />
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-48 p-2" align="center">
        <div className="space-y-2">
          {/* Native color picker */}
          <Input
            type="color"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="h-8 w-full cursor-pointer p-0 border-0"
          />
          {/* Preset colors */}
          <div className="grid grid-cols-5 gap-1">
            {presets.map((color) => (
              <button
                key={color}
                type="button"
                className={cn(
                  'h-6 w-6 rounded-md border-2 transition-all hover:scale-110',
                  value === color
                    ? 'border-foreground'
                    : 'border-transparent hover:border-border'
                )}
                style={{ backgroundColor: color }}
                onClick={() => handlePresetClick(color)}
                title={color}
              >
                {value === color && (
                  <Check className="h-3 w-3 mx-auto text-white drop-shadow-md" />
                )}
              </button>
            ))}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function ColorPickerInline({
  upColor,
  downColor,
  onUpChange,
  onDownChange,
  onReset,
  className,
}: ColorPickerInlineProps) {
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <ColorDot
        value={upColor}
        onChange={onUpChange}
        icon={TrendingUp}
        presets={['#22c55e', '#3b82f6', '#14b8a6', '#8b5cf6', '#10b981']}
        title="Up color"
      />
      <ColorDot
        value={downColor}
        onChange={onDownChange}
        icon={TrendingDown}
        presets={['#ef4444', '#f97316', '#ec4899', '#dc2626', '#f43f5e']}
        title="Down color"
      />
      {onReset && (
        <button
          onClick={onReset}
          className="text-muted-foreground hover:text-foreground transition-colors"
          title="Reset colors"
        >
          <RotateCcw className="h-3 w-3" />
        </button>
      )}
    </div>
  );
}
