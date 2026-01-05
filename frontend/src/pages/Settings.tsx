import { Outlet, NavLink, useLocation, Navigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { useSEO } from '@/lib/seo';
import {
  User,
  Shield,
  Palette,
  Bell,
  Key,
  Link2,
  Lock,
} from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

const settingsNavItems = [
  {
    title: 'Profile',
    href: '/settings/profile',
    icon: User,
    description: 'Manage your account details',
  },
  {
    title: 'Security',
    href: '/settings/security',
    icon: Shield,
    description: 'Password and two-factor authentication',
  },
  {
    title: 'Appearance',
    href: '/settings/appearance',
    icon: Palette,
    description: 'Themes and color preferences',
  },
  {
    title: 'Notifications',
    href: '/settings/notifications',
    icon: Bell,
    description: 'Alert channels and rules',
  },
  {
    title: 'API Keys',
    href: '/settings/api-keys',
    icon: Key,
    description: 'Manage your API access',
  },
  {
    title: 'Connected Accounts',
    href: '/settings/connections',
    icon: Link2,
    description: 'Linked social accounts',
  },
  {
    title: 'Privacy',
    href: '/settings/privacy',
    icon: Lock,
    description: 'Profile visibility and data',
  },
];

function SettingsNav() {
  return (
    <nav className="flex flex-col gap-1">
      {settingsNavItems.map((item) => (
        <NavLink
          key={item.href}
          to={item.href}
          className={({ isActive }) =>
            cn(
              'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
              'hover:bg-accent hover:text-accent-foreground',
              isActive
                ? 'bg-accent text-accent-foreground font-medium'
                : 'text-muted-foreground'
            )
          }
        >
          <item.icon className="h-4 w-4" />
          <span>{item.title}</span>
        </NavLink>
      ))}
    </nav>
  );
}

export function SettingsLayout() {
  useSEO({
    title: 'Settings',
    description: 'Manage your StonkMarket account settings',
    noindex: true,
  });

  const location = useLocation();

  // Redirect /settings to /settings/profile
  if (location.pathname === '/settings') {
    return <Navigate to="/settings/profile" replace />;
  }

  return (
    <div className="container max-w-6xl py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-muted-foreground">
          Manage your account and preferences
        </p>
      </div>

      <div className="flex flex-col gap-6 lg:flex-row">
        {/* Sidebar Navigation */}
        <aside className="lg:w-64 shrink-0">
          <ScrollArea className="h-auto lg:h-[calc(100vh-200px)]">
            <SettingsNav />
          </ScrollArea>
        </aside>

        <Separator orientation="vertical" className="hidden lg:block" />

        {/* Content Area */}
        <main className="flex-1 min-w-0">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default SettingsLayout;
