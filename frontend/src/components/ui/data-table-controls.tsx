/* eslint-disable react-refresh/only-export-components */
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, Search, ChevronsLeft, ChevronsRight } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface DataTableControlsProps {
  // Search
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  searchPlaceholder?: string;
  
  // Pagination
  currentPage: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange?: (size: number) => void;
  pageSizeOptions?: number[];
  
  // Optional: display info
  itemName?: string; // e.g., "symbols", "tasks"
}

export function DataTableControls({
  searchValue,
  onSearchChange,
  searchPlaceholder = 'Search...',
  currentPage,
  totalPages,
  totalItems,
  pageSize,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [10, 25, 50, 100],
  itemName = 'items',
}: DataTableControlsProps) {
  const startItem = totalItems === 0 ? 0 : (currentPage - 1) * pageSize + 1;
  const endItem = Math.min(currentPage * pageSize, totalItems);

  return (
    <div className="space-y-4">
      {/* Search bar */}
      {onSearchChange && (
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={searchPlaceholder}
            value={searchValue || ''}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-9"
          />
        </div>
      )}

      {/* Pagination controls */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Left: Items info */}
        <div className="text-sm text-muted-foreground">
          {totalItems > 0 ? (
            <>
              Showing <strong className="text-foreground">{startItem}</strong>-
              <strong className="text-foreground">{endItem}</strong> of{' '}
              <strong className="text-foreground">{totalItems}</strong> {itemName}
            </>
          ) : (
            <>No {itemName} found</>
          )}
        </div>

        {/* Right: Page controls */}
        <div className="flex items-center gap-2">
          {/* Page size selector */}
          {onPageSizeChange && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Show</span>
              <Select
                value={pageSize.toString()}
                onValueChange={(value) => onPageSizeChange(parseInt(value))}
              >
                <SelectTrigger className="h-8 w-[70px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {pageSizeOptions.map((size) => (
                    <SelectItem key={size} value={size.toString()}>
                      {size}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Page navigation */}
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => onPageChange(1)}
              disabled={currentPage <= 1}
            >
              <ChevronsLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage <= 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>

            <span className="px-3 text-sm text-muted-foreground">
              Page <strong className="text-foreground">{currentPage}</strong> of{' '}
              <strong className="text-foreground">{Math.max(1, totalPages)}</strong>
            </span>

            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage >= totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => onPageChange(totalPages)}
              disabled={currentPage >= totalPages}
            >
              <ChevronsRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Hook for pagination logic
export function usePagination<T>(
  items: T[],
  initialPageSize: number = 10
) {
  const [currentPage, setCurrentPage] = React.useState(1);
  const [pageSize, setPageSize] = React.useState(initialPageSize);
  const [searchValue, setSearchValue] = React.useState('');

  const totalItems = items.length;
  const totalPages = Math.ceil(totalItems / pageSize);
  
  // Reset to page 1 when items change significantly
  React.useEffect(() => {
    if (currentPage > Math.ceil(items.length / pageSize)) {
      setCurrentPage(1);
    }
  }, [items.length, pageSize, currentPage]);

  const paginatedItems = React.useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return items.slice(start, start + pageSize);
  }, [items, currentPage, pageSize]);

  const handlePageChange = (page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  };

  const handlePageSizeChange = (size: number) => {
    setPageSize(size);
    setCurrentPage(1); // Reset to first page
  };

  return {
    currentPage,
    pageSize,
    totalPages,
    totalItems,
    paginatedItems,
    searchValue,
    setSearchValue,
    onPageChange: handlePageChange,
    onPageSizeChange: handlePageSizeChange,
  };
}

import * as React from 'react';
