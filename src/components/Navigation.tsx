import { Button } from "./ui/button";
import { Moon, Sun, Menu } from "lucide-react";
import { useState } from "react";

export function Navigation() {
  const [isDark, setIsDark] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-teal-500 rounded-xl flex items-center justify-center">
              <span className="text-white text-xl">👋</span>
            </div>
            <span className="text-gray-900">SignSpeak</span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-gray-600 hover:text-gray-900 transition-colors">
              Features
            </a>
            <a href="#learn" className="text-gray-600 hover:text-gray-900 transition-colors">
              Learn
            </a>
            <a href="#practice" className="text-gray-600 hover:text-gray-900 transition-colors">
              Practice
            </a>
            <a href="#about" className="text-gray-600 hover:text-gray-900 transition-colors">
              About
            </a>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsDark(!isDark)}
              className="hidden md:flex p-2 rounded-lg hover:bg-gray-100 transition-colors"
              aria-label="Toggle dark mode"
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <Button variant="ghost" className="hidden md:inline-flex">
              Log In
            </Button>
            <Button className="bg-blue-600 hover:bg-blue-700">
              Sign Up
            </Button>
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-gray-100"
              aria-label="Toggle menu"
            >
              <Menu className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-gray-200">
            <div className="flex flex-col gap-4">
              <a href="#features" className="text-gray-600 hover:text-gray-900">
                Features
              </a>
              <a href="#learn" className="text-gray-600 hover:text-gray-900">
                Learn
              </a>
              <a href="#practice" className="text-gray-600 hover:text-gray-900">
                Practice
              </a>
              <a href="#about" className="text-gray-600 hover:text-gray-900">
                About
              </a>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
