import { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  useWatchlists,
  useWatchlistDetail,
  useDippingStocks,
  useOpportunities,
} from '@/features/watchlist/api/queries';
import {
  useCreateWatchlist,
  useUpdateWatchlist,
  useDeleteWatchlist,
  useAddWatchlistItem,
  useUpdateWatchlistItem,
  useDeleteWatchlistItem,
} from '@/features/watchlist/api/mutations';
import type { Watchlist, WatchlistItem } from '@/features/watchlist/api/types';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { 
  Plus, 
  Trash2, 
  MoreHorizontal, 
  Pencil, 
  Eye,
  TrendingDown,
  Target,
  AlertTriangle,
  Star,
  Bell,
  ExternalLink,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.05 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

// Skeleton components
function TableSkeleton() {
  return (
    <div className="space-y-2">
      {[...Array(5)].map((_, i) => (
        <Skeleton key={i} className="h-12 w-full" />
      ))}
    </div>
  );
}

// Format helpers
function formatPrice(price: number | null | undefined): string {
  if (price == null) return '-';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(price);
}

function formatPercent(value: number | null | undefined): string {
  if (value == null) return '-';
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
}

export function WatchlistPage() {
  const [selectedWatchlistId, setSelectedWatchlistId] = useState<number | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showAddSymbolDialog, setShowAddSymbolDialog] = useState(false);
  const [showEditItemDialog, setShowEditItemDialog] = useState(false);
  const [editingWatchlist, setEditingWatchlist] = useState<Watchlist | null>(null);
  const [editingItem, setEditingItem] = useState<WatchlistItem | null>(null);
  
  // Form state
  const [newWatchlistName, setNewWatchlistName] = useState('');
  const [newWatchlistDescription, setNewWatchlistDescription] = useState('');
  const [newWatchlistIsDefault, setNewWatchlistIsDefault] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [newTargetPrice, setNewTargetPrice] = useState('');
  const [newNotes, setNewNotes] = useState('');
  const [newAlertOnDip, setNewAlertOnDip] = useState(true);
  
  // Queries
  const { data: watchlists, isLoading: watchlistsLoading } = useWatchlists();
  const { data: watchlistDetail, isLoading: detailLoading } = useWatchlistDetail(selectedWatchlistId);
  const { data: dippingStocks } = useDippingStocks(10);
  const { data: opportunities } = useOpportunities();
  
  // Mutations
  const createWatchlist = useCreateWatchlist();
  const updateWatchlist = useUpdateWatchlist();
  const deleteWatchlist = useDeleteWatchlist();
  const addItem = useAddWatchlistItem();
  const updateItem = useUpdateWatchlistItem();
  const deleteItem = useDeleteWatchlistItem();
  
  // Auto-select first watchlist on initial load
  const effectiveWatchlistId = selectedWatchlistId ?? 
    (watchlists?.find(w => w.is_default)?.id ?? watchlists?.[0]?.id ?? null);
  
  // Sync local state with derived value (only on first load)
  if (effectiveWatchlistId !== null && selectedWatchlistId === null && watchlists?.length) {
    setSelectedWatchlistId(effectiveWatchlistId);
  }
  
  // Handlers - React Compiler handles memoization, no useCallback needed
  async function handleCreateWatchlist() {
    if (!newWatchlistName.trim()) return;
    
    await createWatchlist.mutateAsync({
      name: newWatchlistName.trim(),
      description: newWatchlistDescription.trim() || undefined,
      is_default: newWatchlistIsDefault,
    });
    
    setShowCreateDialog(false);
    setNewWatchlistName('');
    setNewWatchlistDescription('');
    setNewWatchlistIsDefault(false);
  }
  
  async function handleUpdateWatchlist() {
    if (!editingWatchlist || !newWatchlistName.trim()) return;
    
    await updateWatchlist.mutateAsync({
      id: editingWatchlist.id,
      data: {
        name: newWatchlistName.trim(),
        description: newWatchlistDescription.trim() || null,
        is_default: newWatchlistIsDefault,
      },
    });
    
    setShowEditDialog(false);
    setEditingWatchlist(null);
  }
  
  async function handleDeleteWatchlist(id: number) {
    if (!confirm('Are you sure you want to delete this watchlist?')) return;
    
    await deleteWatchlist.mutateAsync(id);
    if (selectedWatchlistId === id) {
      setSelectedWatchlistId(null);
    }
  }
  
  async function handleAddSymbol() {
    if (!selectedWatchlistId || !newSymbol.trim()) return;
    
    await addItem.mutateAsync({
      watchlistId: selectedWatchlistId,
      data: {
        symbol: newSymbol.trim().toUpperCase(),
        target_price: newTargetPrice ? parseFloat(newTargetPrice) : undefined,
        notes: newNotes.trim() || undefined,
        alert_on_dip: newAlertOnDip,
      },
    });
    
    setShowAddSymbolDialog(false);
    setNewSymbol('');
    setNewTargetPrice('');
    setNewNotes('');
    setNewAlertOnDip(true);
  }
  
  async function handleUpdateItem() {
    if (!selectedWatchlistId || !editingItem) return;
    
    await updateItem.mutateAsync({
      watchlistId: selectedWatchlistId,
      itemId: editingItem.id,
      data: {
        target_price: newTargetPrice ? parseFloat(newTargetPrice) : null,
        notes: newNotes.trim() || null,
        alert_on_dip: newAlertOnDip,
      },
    });
    
    setShowEditItemDialog(false);
    setEditingItem(null);
  }
  
  async function handleDeleteItem(itemId: number) {
    if (!selectedWatchlistId) return;
    await deleteItem.mutateAsync({ watchlistId: selectedWatchlistId, itemId });
  }
  
  function openEditDialog(wl: Watchlist) {
    setEditingWatchlist(wl);
    setNewWatchlistName(wl.name);
    setNewWatchlistDescription(wl.description || '');
    setNewWatchlistIsDefault(wl.is_default);
    setShowEditDialog(true);
  }
  
  function openEditItemDialog(wlItem: WatchlistItem) {
    setEditingItem(wlItem);
    setNewTargetPrice(wlItem.target_price?.toString() || '');
    setNewNotes(wlItem.notes || '');
    setNewAlertOnDip(wlItem.alert_on_dip);
    setShowEditItemDialog(true);
  }
  
  // Derived state - no useMemo needed, React Compiler optimizes this
  const selectedWatchlist = watchlists?.find(w => w.id === selectedWatchlistId);
  
  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Watchlist</h1>
          <p className="text-muted-foreground">
            Track stocks and get alerts when they hit your target prices
          </p>
        </div>
        <Button onClick={() => setShowCreateDialog(true)}>
          <Plus className="h-4 w-4 mr-2" />
          New Watchlist
        </Button>
      </div>
      
      {/* Quick Stats Cards */}
      <motion.div 
        className="grid gap-4 md:grid-cols-3"
        variants={container}
        initial="hidden"
        animate="show"
      >
        <motion.div variants={item}>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <Eye className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Watchlists</p>
                  <p className="text-2xl font-bold">
                    {watchlistsLoading ? (
                      <Skeleton className="h-8 w-8" />
                    ) : (
                      watchlists?.length ?? 0
                    )}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        
        <motion.div variants={item}>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-orange-500/10">
                  <TrendingDown className="h-5 w-5 text-orange-500" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Stocks Dipping</p>
                  <p className="text-2xl font-bold text-orange-500">
                    {dippingStocks?.length ?? 0}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        
        <motion.div variants={item}>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-green-500/10">
                  <Target className="h-5 w-5 text-green-500" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">At Target Price</p>
                  <p className="text-2xl font-bold text-green-500">
                    {opportunities?.length ?? 0}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
      
      {/* Alerts Section */}
      {(dippingStocks?.length ?? 0) > 0 && (
        <Card className="border-orange-500/50 bg-orange-500/5">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-orange-500">
              <AlertTriangle className="h-5 w-5" />
              Dipping Stocks Alert
            </CardTitle>
            <CardDescription>
              These watched stocks are currently dipping 10% or more from recent highs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {dippingStocks?.map(stock => (
                <Link key={stock.symbol} to={`/stock/${stock.symbol}`}>
                  <Badge variant="outline" className="cursor-pointer hover:bg-orange-500/10">
                    <span className="font-medium">{stock.symbol}</span>
                    <span className="ml-2 text-orange-500">{formatPercent(-(stock.dip_percent ?? 0))}</span>
                  </Badge>
                </Link>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Opportunities Section */}
      {(opportunities?.length ?? 0) > 0 && (
        <Card className="border-green-500/50 bg-green-500/5">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-green-500">
              <Target className="h-5 w-5" />
              Target Price Opportunities
            </CardTitle>
            <CardDescription>
              These stocks have hit or gone below your target buy price
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {opportunities?.map(opp => (
                <Link key={opp.symbol} to={`/stock/${opp.symbol}`}>
                  <Badge variant="outline" className="cursor-pointer hover:bg-green-500/10">
                    <span className="font-medium">{opp.symbol}</span>
                    <span className="ml-2">
                      {formatPrice(opp.current_price)} 
                      <span className="text-green-500 ml-1">
                        ({formatPercent(-opp.discount_percent)} below target)
                      </span>
                    </span>
                  </Badge>
                </Link>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
        {/* Watchlist Selector */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Your Watchlists</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {watchlistsLoading ? (
              <div className="space-y-2">
                {[...Array(3)].map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : watchlists?.length === 0 ? (
              <p className="text-muted-foreground text-sm py-4 text-center">
                No watchlists yet. Create one to start tracking stocks.
              </p>
            ) : (
              watchlists?.map(wl => (
                <div
                  key={wl.id}
                  className={cn(
                    "flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors",
                    selectedWatchlistId === wl.id 
                      ? "bg-primary/10 border border-primary/20" 
                      : "hover:bg-muted"
                  )}
                  onClick={() => setSelectedWatchlistId(wl.id)}
                >
                  <div className="flex items-center gap-2">
                    {wl.is_default && (
                      <Star className="h-4 w-4 text-yellow-500 fill-yellow-500" />
                    )}
                    <div>
                      <p className="font-medium">{wl.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {wl.item_count} {wl.item_count === 1 ? 'stock' : 'stocks'}
                      </p>
                    </div>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); openEditDialog(wl); }}>
                        <Pencil className="h-4 w-4 mr-2" />
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuItem 
                        onClick={(e) => { e.stopPropagation(); handleDeleteWatchlist(wl.id); }}
                        className="text-destructive"
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              ))
            )}
          </CardContent>
        </Card>
        
        {/* Watchlist Items */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>{selectedWatchlist?.name ?? 'Select a Watchlist'}</CardTitle>
              {selectedWatchlist?.description && (
                <CardDescription>{selectedWatchlist.description}</CardDescription>
              )}
            </div>
            {selectedWatchlistId && (
              <Button onClick={() => setShowAddSymbolDialog(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Add Stock
              </Button>
            )}
          </CardHeader>
          <CardContent>
            {detailLoading ? (
              <TableSkeleton />
            ) : !selectedWatchlistId ? (
              <p className="text-muted-foreground text-center py-8">
                Select or create a watchlist to view stocks
              </p>
            ) : !watchlistDetail?.items?.length ? (
              <div className="text-center py-8">
                <Eye className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
                <p className="text-muted-foreground mb-4">
                  This watchlist is empty. Add some stocks to track.
                </p>
                <Button onClick={() => setShowAddSymbolDialog(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  Add Your First Stock
                </Button>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead className="text-right">Current Price</TableHead>
                    <TableHead className="text-right">Dip %</TableHead>
                    <TableHead className="text-right">Target Price</TableHead>
                    <TableHead>Alert</TableHead>
                    <TableHead className="w-[50px]"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <AnimatePresence>
                    {watchlistDetail.items.map(wlItem => {
                      const atTarget = wlItem.target_price && wlItem.current_price 
                        && wlItem.current_price <= wlItem.target_price;
                      const isDipping = (wlItem.dip_percent ?? 0) >= 10;
                      
                      return (
                        <motion.tr
                          key={wlItem.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 20 }}
                          className={cn(
                            atTarget && "bg-green-500/5",
                            isDipping && !atTarget && "bg-orange-500/5"
                          )}
                        >
                          <TableCell>
                            <Link 
                              to={`/stock/${wlItem.symbol}`}
                              className="flex items-center gap-2 hover:text-primary transition-colors"
                            >
                              <span className="font-medium">{wlItem.symbol}</span>
                              <ExternalLink className="h-3 w-3" />
                            </Link>
                            {wlItem.notes && (
                              <p className="text-xs text-muted-foreground mt-1">
                                {wlItem.notes}
                              </p>
                            )}
                          </TableCell>
                          <TableCell className="text-right font-mono">
                            {formatPrice(wlItem.current_price)}
                          </TableCell>
                          <TableCell className="text-right">
                            {wlItem.dip_percent != null ? (
                              <Badge 
                                variant="outline"
                                className={cn(
                                  isDipping ? "text-orange-500 border-orange-500/50" : ""
                                )}
                              >
                                <TrendingDown className="h-3 w-3 mr-1" />
                                {wlItem.dip_percent.toFixed(1)}%
                              </Badge>
                            ) : '-'}
                          </TableCell>
                          <TableCell className="text-right font-mono">
                            {wlItem.target_price ? (
                              <span className={cn(atTarget && "text-green-500 font-semibold")}>
                                {formatPrice(wlItem.target_price)}
                                {atTarget && <Target className="h-3 w-3 inline ml-1" />}
                              </span>
                            ) : '-'}
                          </TableCell>
                          <TableCell>
                            {wlItem.alert_on_dip && (
                              <Bell className="h-4 w-4 text-muted-foreground" />
                            )}
                          </TableCell>
                          <TableCell>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-8 w-8">
                                  <MoreHorizontal className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem onClick={() => openEditItemDialog(wlItem)}>
                                  <Pencil className="h-4 w-4 mr-2" />
                                  Edit
                                </DropdownMenuItem>
                                <DropdownMenuItem 
                                  onClick={() => handleDeleteItem(wlItem.id)}
                                  className="text-destructive"
                                >
                                  <Trash2 className="h-4 w-4 mr-2" />
                                  Remove
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </TableCell>
                        </motion.tr>
                      );
                    })}
                  </AnimatePresence>
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
      
      {/* Create Watchlist Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Watchlist</DialogTitle>
            <DialogDescription>
              Create a new watchlist to track stocks you're interested in.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={newWatchlistName}
                onChange={(e) => setNewWatchlistName(e.target.value)}
                placeholder="e.g., Tech Stocks, Dividend Plays"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="description">Description (optional)</Label>
              <Input
                id="description"
                value={newWatchlistDescription}
                onChange={(e) => setNewWatchlistDescription(e.target.value)}
                placeholder="What's this watchlist for?"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="default">Set as default</Label>
                <p className="text-xs text-muted-foreground">
                  This will be your primary watchlist
                </p>
              </div>
              <Switch
                id="default"
                checked={newWatchlistIsDefault}
                onCheckedChange={setNewWatchlistIsDefault}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleCreateWatchlist}
              disabled={!newWatchlistName.trim() || createWatchlist.isPending}
            >
              {createWatchlist.isPending ? 'Creating...' : 'Create Watchlist'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Edit Watchlist Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Watchlist</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="edit-name">Name</Label>
              <Input
                id="edit-name"
                value={newWatchlistName}
                onChange={(e) => setNewWatchlistName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-description">Description</Label>
              <Input
                id="edit-description"
                value={newWatchlistDescription}
                onChange={(e) => setNewWatchlistDescription(e.target.value)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="edit-default">Set as default</Label>
              </div>
              <Switch
                id="edit-default"
                checked={newWatchlistIsDefault}
                onCheckedChange={setNewWatchlistIsDefault}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowEditDialog(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleUpdateWatchlist}
              disabled={!newWatchlistName.trim() || updateWatchlist.isPending}
            >
              {updateWatchlist.isPending ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Add Symbol Dialog */}
      <Dialog open={showAddSymbolDialog} onOpenChange={setShowAddSymbolDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Stock to Watchlist</DialogTitle>
            <DialogDescription>
              Add a stock symbol to track in your watchlist.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="symbol">Stock Symbol</Label>
              <Input
                id="symbol"
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                placeholder="e.g., AAPL, MSFT, GOOGL"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="target-price">Target Price (optional)</Label>
              <Input
                id="target-price"
                type="number"
                step="0.01"
                value={newTargetPrice}
                onChange={(e) => setNewTargetPrice(e.target.value)}
                placeholder="Price at which you'd like to buy"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="notes">Notes (optional)</Label>
              <Input
                id="notes"
                value={newNotes}
                onChange={(e) => setNewNotes(e.target.value)}
                placeholder="Why are you watching this stock?"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="alert">Alert on dip</Label>
                <p className="text-xs text-muted-foreground">
                  Get notified when this stock dips significantly
                </p>
              </div>
              <Switch
                id="alert"
                checked={newAlertOnDip}
                onCheckedChange={setNewAlertOnDip}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddSymbolDialog(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleAddSymbol}
              disabled={!newSymbol.trim() || addItem.isPending}
            >
              {addItem.isPending ? 'Adding...' : 'Add Stock'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Edit Item Dialog */}
      <Dialog open={showEditItemDialog} onOpenChange={setShowEditItemDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit {editingItem?.symbol}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="edit-target-price">Target Price</Label>
              <Input
                id="edit-target-price"
                type="number"
                step="0.01"
                value={newTargetPrice}
                onChange={(e) => setNewTargetPrice(e.target.value)}
                placeholder="Price at which you'd like to buy"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-notes">Notes</Label>
              <Input
                id="edit-notes"
                value={newNotes}
                onChange={(e) => setNewNotes(e.target.value)}
              />
            </div>
            <div className="flex items-center justify-between">
              <Label htmlFor="edit-alert">Alert on dip</Label>
              <Switch
                id="edit-alert"
                checked={newAlertOnDip}
                onCheckedChange={setNewAlertOnDip}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowEditItemDialog(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleUpdateItem}
              disabled={updateItem.isPending}
            >
              {updateItem.isPending ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
