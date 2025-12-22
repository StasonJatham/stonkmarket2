import type { BenchmarkType } from '@/services/api';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { LineChart, X, Check } from 'lucide-react';

interface BenchmarkSelectorProps {
  value: BenchmarkType;
  onChange: (benchmark: BenchmarkType) => void;
  isLoading?: boolean;
  className?: string;
}

const benchmarks: { value: Exclude<BenchmarkType, null>; label: string; description: string }[] = [
  { value: 'SP500', label: 'S&P 500', description: 'US Large Cap Index' },
  { value: 'MSCI_WORLD', label: 'MSCI World', description: 'Global Developed Markets' },
];

export function BenchmarkSelector({ value, onChange, isLoading, className }: BenchmarkSelectorProps) {
  const selectedBenchmark = benchmarks.find(b => b.value === value);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button 
          variant={value ? 'default' : 'outline'} 
          size="sm" 
          className={`h-9 ${className || ''}`}
          disabled={isLoading}
        >
          <LineChart className="h-4 w-4 mr-2" />
          {value ? (
            <span className="flex items-center gap-2">
              <span className="hidden sm:inline">vs</span> {selectedBenchmark?.label}
            </span>
          ) : (
            <>
              <span className="hidden sm:inline">Compare</span>
              <span className="sm:hidden">Benchmark</span>
            </>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>Compare with Benchmark</DropdownMenuLabel>
        <DropdownMenuSeparator />
        
        {benchmarks.map((benchmark) => (
          <DropdownMenuItem
            key={benchmark.value}
            onClick={() => onChange(value === benchmark.value ? null : benchmark.value)}
            className="flex items-center justify-between"
          >
            <div>
              <p className="font-medium">{benchmark.label}</p>
              <p className="text-xs text-muted-foreground">{benchmark.description}</p>
            </div>
            {value === benchmark.value && (
              <Check className="h-4 w-4 text-primary" />
            )}
          </DropdownMenuItem>
        ))}
        
        {value && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => onChange(null)}
              className="text-muted-foreground"
            >
              <X className="h-4 w-4 mr-2" />
              Clear benchmark
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
