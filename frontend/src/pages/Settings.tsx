import { Settings as SettingsIcon } from 'lucide-react';

export default function Settings() {
  return (
    <div className="space-y-6 max-w-2xl">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-gray-900">Settings</h1>
        <p className="mt-1 text-sm text-gray-500">
          Application preferences and configuration.
        </p>
      </div>

      {/* Preferences card */}
      <section
        aria-labelledby="preferences-heading"
        className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden"
      >
        <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-100">
          <SettingsIcon size={18} className="text-gray-400" aria-hidden="true" />
          <h2 id="preferences-heading" className="text-base font-semibold text-gray-900">
            Application Preferences
          </h2>
        </div>

        <div className="px-5 py-6 space-y-4">
          {/* API URL display */}
          <div>
            <label htmlFor="api-url" className="block text-sm font-medium text-gray-700">
              Backend API URL
            </label>
            <p className="mt-0.5 text-xs text-gray-400">
              Configured via the VITE_API_BASE_URL environment variable.
            </p>
            <input
              id="api-url"
              type="text"
              readOnly
              value={
                (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
                'http://localhost:8000/api (default)'
              }
              className="mt-2 w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-500 cursor-not-allowed"
              aria-readonly="true"
            />
          </div>

          {/* Placeholder notice */}
          <div className="rounded-lg border border-dashed border-gray-200 bg-gray-50 px-4 py-4">
            <p className="text-sm text-gray-500">
              Additional configuration options will appear here in future steps, including
              model preferences, display options, and export settings.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
