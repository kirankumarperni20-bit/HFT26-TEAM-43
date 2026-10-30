import { Search, Menu, User } from 'lucide-react';

interface HeaderProps {
  onMenuClick: () => void;
}

export default function Header({ onMenuClick }: HeaderProps) {
  return (
    <header className="sticky top-0 z-10 flex h-14 items-center gap-3 border-b border-gray-200 bg-white px-4 sm:px-6">
      {/* Mobile hamburger */}
      <button
        className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 lg:hidden focus-visible:outline-2 focus-visible:outline-blue-500"
        onClick={onMenuClick}
        aria-label="Open navigation menu"
      >
        <Menu size={20} />
      </button>

      {/* Search */}
      <div className="relative flex-1 max-w-lg">
        <Search
          size={16}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
          aria-hidden="true"
        />
        <input
          id="global-search"
          type="search"
          placeholder="Search documents, topics, or research..."
          className="w-full rounded-lg border border-gray-200 bg-gray-50 py-2 pl-9 pr-4 text-sm text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20"
          aria-label="Search documents, topics, or research"
        />
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Profile placeholder */}
      <div
        className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-600"
        aria-label="User profile"
        role="img"
      >
        <User size={16} />
      </div>
    </header>
  );
}
