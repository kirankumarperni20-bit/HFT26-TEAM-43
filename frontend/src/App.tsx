import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from '@/components/layout/AppLayout';
import Dashboard from '@/pages/Dashboard';
import Documents from '@/pages/Documents';
import DocumentDetails from '@/pages/DocumentDetails';
import ResearchWorkspace from '@/pages/ResearchWorkspace';
import Settings from '@/pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="documents" element={<Documents />} />
          <Route path="documents/:paperId" element={<DocumentDetails />} />
          <Route path="workspace" element={<ResearchWorkspace />} />
          <Route path="settings" element={<Settings />} />
          {/* Catch-all redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
