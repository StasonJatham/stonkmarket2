import { useState, useMemo } from 'react';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { AlertCircle, CheckCircle } from 'lucide-react';

interface CronBuilderProps {
  value: string;
  onChange: (value: string) => void;
}

// Presets for quick selection
const PRESETS = [
  { label: 'Every minute', value: '* * * * *' },
  { label: 'Every 5 minutes', value: '*/5 * * * *' },
  { label: 'Every 15 minutes', value: '*/15 * * * *' },
  { label: 'Every 30 minutes', value: '*/30 * * * *' },
  { label: 'Hourly', value: '0 * * * *' },
  { label: 'Every 2 hours', value: '0 */2 * * *' },
  { label: 'Every 4 hours', value: '0 */4 * * *' },
  { label: 'Every 6 hours', value: '0 */6 * * *' },
  { label: 'Daily at midnight', value: '0 0 * * *' },
  { label: 'Daily at 6 AM', value: '0 6 * * *' },
  { label: 'Daily at 9 AM', value: '0 9 * * *' },
  { label: 'Daily at noon', value: '0 12 * * *' },
  { label: 'Daily at 6 PM', value: '0 18 * * *' },
  { label: 'Weekdays at 9 AM', value: '0 9 * * 1-5' },
  { label: 'Weekly (Sunday midnight)', value: '0 0 * * 0' },
  { label: 'Monthly (1st at midnight)', value: '0 0 1 * *' },
  { label: 'Custom', value: 'custom' },
];

// Validation
function validateCronPart(value: string, min: number, max: number): boolean {
  if (value === '*') return true;
  if (/^\*\/\d+$/.test(value)) {
    const step = parseInt(value.slice(2));
    return step >= 1 && step <= max;
  }
  if (/^\d+-\d+$/.test(value)) {
    const [start, end] = value.split('-').map(Number);
    return start >= min && end <= max && start <= end;
  }
  if (/^[\d,]+$/.test(value)) {
    const nums = value.split(',').map(Number);
    return nums.every(n => n >= min && n <= max);
  }
  if (/^\d+$/.test(value)) {
    const num = parseInt(value);
    return num >= min && num <= max;
  }
  return false;
}

function validateCron(cron: string): { valid: boolean; error?: string } {
  const parts = cron.trim().split(/\s+/);
  if (parts.length !== 5) {
    return { valid: false, error: 'Must have 5 parts: minute hour day month weekday' };
  }
  const [minute, hour, dayOfMonth, month, dayOfWeek] = parts;
  if (!validateCronPart(minute, 0, 59)) return { valid: false, error: 'Invalid minute (0-59)' };
  if (!validateCronPart(hour, 0, 23)) return { valid: false, error: 'Invalid hour (0-23)' };
  if (!validateCronPart(dayOfMonth, 1, 31)) return { valid: false, error: 'Invalid day (1-31)' };
  if (!validateCronPart(month, 1, 12)) return { valid: false, error: 'Invalid month (1-12)' };
  if (!validateCronPart(dayOfWeek, 0, 6)) return { valid: false, error: 'Invalid weekday (0-6)' };
  return { valid: true };
}

function describeCron(cron: string): string {
  const preset = PRESETS.find(p => p.value === cron && p.value !== 'custom');
  if (preset) return preset.label;
  
  const parts = cron.trim().split(/\s+/);
  if (parts.length !== 5) return 'Invalid expression';
  
  const [minute, hour] = parts;
  
  if (minute === '*' && hour === '*') return 'Every minute';
  if (minute.startsWith('*/')) return `Every ${minute.slice(2)} minutes`;
  if (hour.startsWith('*/')) {
    const m = minute === '0' ? '' : ` at minute ${minute}`;
    return `Every ${hour.slice(2)} hours${m}`;
  }
  if (hour !== '*' && minute !== '*') {
    const h = parseInt(hour);
    const m = parseInt(minute);
    const period = h >= 12 ? 'PM' : 'AM';
    const displayHour = h === 0 ? 12 : h > 12 ? h - 12 : h;
    return `At ${displayHour}:${m.toString().padStart(2, '0')} ${period}`;
  }
  
  return cron;
}

export function CronBuilder({ value, onChange }: CronBuilderProps) {
  const [isCustom, setIsCustom] = useState(() => 
    !PRESETS.some(p => p.value === value && p.value !== 'custom')
  );
  const [customValue, setCustomValue] = useState(value);
  
  const validation = useMemo(() => validateCron(value), [value]);
  const description = useMemo(() => describeCron(value), [value]);
  
  const handlePresetChange = (preset: string) => {
    if (preset === 'custom') {
      setIsCustom(true);
      setCustomValue(value);
    } else {
      setIsCustom(false);
      onChange(preset);
    }
  };
  
  const handleCustomChange = (newValue: string) => {
    setCustomValue(newValue);
    onChange(newValue);
  };
  
  const currentPreset = isCustom ? 'custom' : (PRESETS.find(p => p.value === value)?.value || 'custom');

  return (
    <div className="space-y-4">
      {/* Preset Dropdown */}
      <div className="space-y-2">
        <Label>Schedule</Label>
        <Select value={currentPreset} onValueChange={handlePresetChange}>
          <SelectTrigger>
            <SelectValue placeholder="Select schedule..." />
          </SelectTrigger>
          <SelectContent>
            {PRESETS.map((preset) => (
              <SelectItem key={preset.value} value={preset.value}>
                <span className="flex items-center gap-2">
                  {preset.label}
                  {preset.value !== 'custom' && (
                    <span className="text-muted-foreground font-mono text-xs">
                      {preset.value}
                    </span>
                  )}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Custom Input */}
      {isCustom && (
        <div className="space-y-2">
          <Label>Cron Expression</Label>
          <Input
            value={customValue}
            onChange={(e) => handleCustomChange(e.target.value)}
            placeholder="* * * * *"
            className="font-mono"
          />
          <p className="text-xs text-muted-foreground">
            Format: minute (0-59) hour (0-23) day (1-31) month (1-12) weekday (0-6, Sun=0)
          </p>
        </div>
      )}

      {/* Result */}
      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50 border">
        <div className="space-y-1">
          <div className="font-mono text-sm font-medium">{value}</div>
          <div className="text-xs text-muted-foreground">{description}</div>
        </div>
        {validation.valid ? (
          <Badge variant="outline" className="bg-success/10 text-success border-success/30">
            <CheckCircle className="h-3 w-3 mr-1" />
            Valid
          </Badge>
        ) : (
          <Badge variant="destructive">
            <AlertCircle className="h-3 w-3 mr-1" />
            Invalid
          </Badge>
        )}
      </div>
      
      {!validation.valid && (
        <p className="text-xs text-danger">{validation.error}</p>
      )}
    </div>
  );
}

export { validateCron, describeCron };
