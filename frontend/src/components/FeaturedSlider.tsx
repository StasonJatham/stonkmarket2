import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AreaChart, Area, ResponsiveContainer } from 'recharts';
import type { DipStock, ChartDataPoint } from '@/services/api';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { ChevronLeft, ChevronRight, TrendingDown } from 'lucide-react';

interface FeaturedSliderProps {
  stocks: DipStock[];
  chartDataMap?: Record<string, ChartDataPoint[]>;
  isLoading?: boolean;
  onSelectStock: (symbol: string) => void;
}

function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
}

export function FeaturedSlider({ stocks, chartDataMap, isLoading, onSelectStock }: FeaturedSliderProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [direction, setDirection] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Take top 5 deepest dips
  const featuredStocks = stocks.slice(0, 5);

  const startAutoplay = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      setDirection(1);
      setCurrentIndex((prev) => (prev + 1) % featuredStocks.length);
    }, 5000);
  }, [featuredStocks.length]);

  useEffect(() => {
    if (featuredStocks.length > 1) {
      startAutoplay();
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [featuredStocks.length, startAutoplay]);

  const goTo = (index: number) => {
    setDirection(index > currentIndex ? 1 : -1);
    setCurrentIndex(index);
    startAutoplay();
  };

  const goNext = () => {
    setDirection(1);
    setCurrentIndex((prev) => (prev + 1) % featuredStocks.length);
    startAutoplay();
  };

  const goPrev = () => {
    setDirection(-1);
    setCurrentIndex((prev) => (prev - 1 + featuredStocks.length) % featuredStocks.length);
    startAutoplay();
  };

  if (isLoading) {
    return (
      <Card className="overflow-hidden">
        <CardContent className="p-6">
          <Skeleton className="h-48 w-full rounded-lg" />
        </CardContent>
      </Card>
    );
  }

  if (featuredStocks.length === 0) {
    return null;
  }

  const currentStock = featuredStocks[currentIndex];
  const chartData = chartDataMap?.[currentStock.symbol] || [];
  const miniChartData = chartData.slice(-60).map((p, i) => ({ x: i, y: p.close }));

  const variants = {
    enter: (dir: number) => ({
      x: dir > 0 ? 200 : -200,
      opacity: 0,
      scale: 0.98,
    }),
    center: {
      x: 0,
      opacity: 1,
      scale: 1,
    },
    exit: (dir: number) => ({
      x: dir > 0 ? -200 : 200,
      opacity: 0,
      scale: 0.98,
    }),
  };

  return (
    <Card className="overflow-hidden bg-gradient-to-br from-background to-muted/30">
      <CardContent className="p-0 relative">
        {/* Navigation Arrows */}
        {featuredStocks.length > 1 && (
          <>
            <Button
              variant="ghost"
              size="icon"
              className="absolute left-2 top-1/2 -translate-y-1/2 z-10 h-8 w-8 rounded-full bg-background/80 backdrop-blur"
              onClick={goPrev}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-2 top-1/2 -translate-y-1/2 z-10 h-8 w-8 rounded-full bg-background/80 backdrop-blur"
              onClick={goNext}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </>
        )}

        {/* Slide Content */}
        <div className="relative h-52 overflow-hidden">
          <AnimatePresence initial={false} mode="popLayout" custom={direction}>
            <motion.div
              key={currentStock.symbol}
              custom={direction}
              variants={variants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ 
                x: { type: "spring", stiffness: 300, damping: 30 },
                opacity: { duration: 0.2 }
              }}
              className="absolute inset-0 p-6 cursor-pointer"
              onClick={() => onSelectStock(currentStock.symbol)}
            >
              <div className="flex h-full gap-6">
                {/* Left: Stock Info */}
                <div className="flex-1 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="destructive" className="gap-1">
                        <TrendingDown className="h-3 w-3" />
                        Biggest Dip
                      </Badge>
                    </div>
                    <h2 className="text-3xl font-bold tracking-tight">{currentStock.symbol}</h2>
                    <p className="text-muted-foreground text-sm mt-1 line-clamp-1">
                      {currentStock.name || currentStock.symbol}
                    </p>
                  </div>

                  <div className="flex items-end gap-4">
                    <div>
                      <p className="text-3xl font-bold font-mono">
                        ${currentStock.last_price.toFixed(2)}
                      </p>
                      <p className={`text-sm font-medium ${
                        currentStock.depth < 0 ? 'text-danger' : 'text-success'
                      }`}>
                        {formatPercent(currentStock.depth * 100)} from peak
                      </p>
                    </div>
                    {currentStock.days_since_dip && (
                      <div className="text-sm text-muted-foreground">
                        <span className="font-medium text-foreground">{currentStock.days_since_dip}</span> days in dip
                      </div>
                    )}
                  </div>
                </div>

                {/* Right: Chart */}
                <div className="w-1/2 h-full hidden sm:block">
                  {miniChartData.length > 0 && (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={miniChartData}>
                        <defs>
                          <linearGradient id="featured-gradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="var(--danger)" stopOpacity={0.4} />
                            <stop offset="100%" stopColor="var(--danger)" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <Area
                          type="monotone"
                          dataKey="y"
                          stroke="var(--danger)"
                          strokeWidth={2}
                          fill="url(#featured-gradient)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Dots */}
        {featuredStocks.length > 1 && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-1.5">
            {featuredStocks.map((_, i) => (
              <button
                key={i}
                className={`w-2 h-2 rounded-full transition-all ${
                  i === currentIndex
                    ? 'bg-foreground w-4'
                    : 'bg-foreground/30 hover:bg-foreground/50'
                }`}
                onClick={() => goTo(i)}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
