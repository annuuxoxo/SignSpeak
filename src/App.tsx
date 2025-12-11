import { Navigation } from "./components/Navigation";
import { Hero } from "./components/Hero";
import { LiveTranslation } from "./components/LiveTranslation";
import { LearningMode } from "./components/LearningMode";
import { PracticeMode } from "./components/PracticeMode";
import { Accessibility } from "./components/Accessibility";
import { Testimonials } from "./components/Testimonials";
import { Footer } from "./components/Footer";

export default function App() {
  return (
    <div className="min-h-screen bg-white">
      <Navigation />
      <main>
        <Hero />
        <LiveTranslation />
        <LearningMode />
        <PracticeMode />
        <Accessibility />
        <Testimonials />
      </main>
      <Footer />
    </div>
  );
}
