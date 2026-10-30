import { NavLink } from 'react-router-dom';
import {
  BookOpen,
  LayoutDashboard,
  FileText,
  FlaskConical,
  Settings,
  X,
} from 'lucide-react';

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/documents', label: 'Documents', icon: FileText, end: false },
  { to: '/workspace', label: 'Research Workspace', icon: FlaskConical, end: false },
  { to: '/settings', label: 'Settings', icon: Settings, end: false },
];

export default function Sidebar({ open, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-20 bg-black/40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar panel */}
      <aside
        className={[
          'fixed top-0 left-0 z-30 flex h-full w-64 flex-col bg-[#0f1c2e] transition-transform duration-200',
          'lg:static lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        ].join(' ')}
        aria-label="Application navigation"
      >
        {/* Brand */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-white/10">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/20">
            <BookOpen size={20} className="text-blue-400" />
          </div>
          <span className="text-base font-semibold tracking-tight text-white">
            Research Assistant
          </span>
          {/* Mobile close button */}
          <button
            className="ml-auto rounded p-1 text-white/60 hover:text-white lg:hidden focus-visible:outline-2 focus-visible:outline-blue-400"
            onClick={onClose}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        {/* Nav items */}
        <nav className="flex-1 overflow-y-auto px-3 py-4" aria-label="Primary">
          <ul className="space-y-1" role="list">
            {navItems.map(({ to, label, icon: Icon, end }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={end}
                  onClick={() => {
                    if (window.innerWidth < 1024) onClose();
                  }}
                  className={({ isActive }) =>
                    [
                      'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-blue-500/20 text-blue-300'
                        : 'text-white/60 hover:bg-white/5 hover:text-white',
                    ].join(' ')
                  }
                >
                  <Icon size={18} />
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Footer label */}
        <div className="px-5 py-4 border-t border-white/10">
          <p className="text-xs text-white/30">AI Research Assistant v0.1</p>
        </div>
      </aside>
    </>
  );
}
