
import React, { useState } from 'react';
import { Logo } from './Icons';
import { Tooltip } from './Tooltip';
import {
  Sparkles,
  Settings,
  CircleHelp,
  Github,
  PencilRuler,
  ChevronDown,
  Check
} from 'lucide-react';

export type AppView = 'creation' | 'editor';

interface HeaderProps {
  currentView: AppView;
  setCurrentView: (view: AppView) => void;
  onOpenSettings: () => void;
  onOpenFAQ: () => void;
  t: any;
}

export const Header: React.FC<HeaderProps> = ({ 
  currentView, 
  setCurrentView, 
  onOpenSettings, 
  onOpenFAQ,
  t 
}) => {
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);

  const NavLink = ({ view, label, icon: Icon }: { view: AppView, label: string, icon: any }) => (
      <button
          onClick={() => {
              setCurrentView(view);
          }}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              currentView === view 
              ? 'bg-purple-600/20 text-purple-400 border border-purple-500/20' 
              : 'text-white/60 hover:text-white hover:bg-white/5'
          }`}
      >
          <Icon className="w-4 h-4" />
          {label}
      </button>
  );

  return (
    <header className="w-full backdrop-blur-md sticky top-0 z-50 bg-background-dark/30 border-b border-white/5">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-4 py-3 md:px-8 md:py-4 relative">
        
        {/* Logo & Title - Visible on all devices */}
        <div className="flex items-center gap-2 text-white shrink-0">
          <Logo className="size-8 md:size-10" />
          <h1 className="text-white text-lg md:text-xl font-bold leading-tight tracking-[-0.015em]">{t.appTitle}</h1>
        </div>

        {/* Mobile: View Switcher Dropdown (Centered) */}
        <div className="md:hidden absolute left-1/2 -translate-x-1/2 z-50 select-none">
            <button 
                onClick={() => setIsMobileNavOpen(!isMobileNavOpen)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/10 border border-white/10 backdrop-blur-md text-sm font-medium text-white shadow-lg active:scale-95 transition-all"
            >
                {currentView === 'creation' ? <>
                  <Sparkles className="w-4 h-4" />
                  {t.nav_creation}
                </> : <>
                  <PencilRuler className="w-4 h-4" />
                  {t.nav_editor}
                </>}
                <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${isMobileNavOpen ? 'rotate-180' : ''}`} />
            </button>
            
            {/* Dropdown Menu */}
            {isMobileNavOpen && (
                <>
                    <div 
                        className="fixed inset-0 z-40" 
                        onClick={() => setIsMobileNavOpen(false)} 
                    />
                    <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-48 bg-[#1A1625]/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl p-1.5 flex flex-col gap-1 z-50 animate-in fade-in zoom-in-95 duration-200">
                        <button
                            onClick={() => { setCurrentView('creation'); setIsMobileNavOpen(false); }}
                            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${currentView === 'creation' ? 'bg-purple-600/20 text-purple-400' : 'text-white/70 hover:bg-white/5 hover:text-white'}`}
                        >
                            <Sparkles className="w-4 h-4" />
                            {t.nav_creation}
                            {currentView === 'creation' && <Check className="w-3.5 h-3.5 ml-auto" />}
                        </button>
                        <button
                            onClick={() => { setCurrentView('editor'); setIsMobileNavOpen(false); }}
                            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${currentView === 'editor' ? 'bg-purple-600/20 text-purple-400' : 'text-white/70 hover:bg-white/5 hover:text-white'}`}
                        >
                            <PencilRuler className="w-4 h-4" />
                            {t.nav_editor}
                            {currentView === 'editor' && <Check className="w-3.5 h-3.5 ml-auto" />}
                        </button>
                    </div>
                </>
            )}
        </div>

        {/* Desktop: Navigation (Centered) */}
        <div className="hidden md:flex items-center gap-2 absolute left-1/2 -translate-x-1/2">
            <NavLink view="creation" label={t.nav_creation} icon={Sparkles} />
            <NavLink view="editor" label={t.nav_editor} icon={PencilRuler} />
        </div>
        
        {/* Actions */}
        <div className="flex gap-1 shrink-0">
          <Tooltip content={t.sourceCode} position="bottom">
              <a
                href="https://github.com/Amery2010/peinture"
                className="flex items-center justify-center p-2 rounded-lg text-white/70 hover:text-purple-400 hover:bg-white/10 transition-all active:scale-95"
                target="_blank"
              >
                <Github className="w-5 h-5" />
              </a>
          </Tooltip>

          <Tooltip content={t.help} position="bottom">
              <button
                onClick={onOpenFAQ}
                className="flex items-center justify-center p-2 rounded-lg text-white/70 hover:text-green-400 hover:bg-white/10 transition-all active:scale-95"
              >
                <CircleHelp className="w-5 h-5" />
              </button>
          </Tooltip>

          <Tooltip content={t.settings} position="bottom">
              <button
                onClick={onOpenSettings}
                className="flex items-center justify-center p-2 rounded-lg text-white/70 hover:text-purple-400 hover:bg-white/10 transition-all active:scale-95"
              >
                <Settings className="w-5 h-5" />
              </button>
          </Tooltip>
        </div>
      </div>
    </header>
  );
};
