import type { DipStock } from '@/services/api';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { TrendingDown, BarChart2 } from 'lucide-react';

interface RankingTableProps {
  stocks: DipStock[];
  isLoading?: boolean;
  onSelectStock?: (symbol: string) => void;
  selectedSymbol?: string;
}

function formatMarketCap(value: number | null): string {
  if (!value) return '—';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toFixed(0)}`;
}

export function RankingTable({ stocks, isLoading, onSelectStock, selectedSymbol }: RankingTableProps) {
  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 10 }).map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/50">
            <TableHead className="w-12">#</TableHead>
            <TableHead>Symbol</TableHead>
            <TableHead className="hidden md:table-cell">Name</TableHead>
            <TableHead className="text-right">Price</TableHead>
            <TableHead className="text-right hidden sm:table-cell">Dip</TableHead>
            <TableHead className="text-right">Score</TableHead>
            <TableHead className="text-right hidden lg:table-cell">Market Cap</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {stocks.map((stock, index) => {
            const isSelected = selectedSymbol === stock.symbol;
            const dipPercent = stock.depth * 100;

            return (
              <TableRow
                key={stock.symbol}
                className={`cursor-pointer transition-colors hover:bg-muted/50 ${
                  isSelected ? 'bg-muted' : ''
                }`}
                onClick={() => onSelectStock?.(stock.symbol)}
              >
                <TableCell className="font-medium text-muted-foreground">
                  {index + 1}
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{stock.symbol}</span>
                    {stock.symbol_type === 'index' && (
                      <Badge variant="outline" className="text-xs px-1.5 py-0 h-5 border-chart-2 text-chart-2">
                        <BarChart2 className="h-3 w-3 mr-0.5" />
                        Index
                      </Badge>
                    )}
                  </div>
                </TableCell>
                <TableCell className="hidden md:table-cell text-muted-foreground truncate max-w-48">
                  {stock.name || '—'}
                </TableCell>
                <TableCell className="text-right font-mono">
                  ${stock.last_price.toFixed(2)}
                </TableCell>
                <TableCell className="text-right hidden sm:table-cell">
                  <div className="flex items-center justify-end gap-1">
                    <TrendingDown className="h-4 w-4 text-danger" />
                    <span className="text-danger">
                      -{dipPercent.toFixed(2)}%
                    </span>
                  </div>
                </TableCell>
                <TableCell className="text-right">
                  <span className="inline-flex items-center justify-center w-12 h-6 rounded-full bg-foreground text-background text-sm font-semibold">
                    {(stock.dip_score ?? 0).toFixed(0)}
                  </span>
                </TableCell>
                <TableCell className="text-right hidden lg:table-cell text-muted-foreground font-mono">
                  {formatMarketCap(stock.market_cap)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
