import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Video, Volume2, MessageSquare, Activity } from "lucide-react";
import { Progress } from "./ui/progress";
import Webcam from 'react-webcam';

export function LiveTranslation() {
  return (
    <section id="features" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <Badge className="mb-4 bg-teal-100 text-teal-700 hover:bg-teal-100">
            Real-Time Translation
          </Badge>
          <h2 className="text-gray-900 mb-4">
            See Your Signs Come to Life
          </h2>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Our AI-powered system instantly recognizes your gestures and converts them into text and speech with industry-leading accuracy.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
          {/* Main Translation Card */}
          <Card className="p-6 shadow-xl border-2 border-blue-100">
            <div className="space-y-4">
              {/* Webcam Preview */}
              <div className="relative bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl aspect-video overflow-hidden">
                <div className="absolute inset-0 flex items-center justify-center webcam-mirror">
                  <Webcam classname="webcam-mirror" />
                </div>
                
                {/* Live Indicator */}
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 text-white px-3 py-1 rounded-full text-sm">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                  LIVE
                </div>

                {/* Hand Detection Overlay */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative">
                    <div className="text-6xl"></div>
                    {/* Detection Frame */}
                    <div className="absolute inset-0 border-4 border-teal-400 rounded-lg animate-pulse"></div>
                  </div>
                </div>

                {/* Status Bar */}
                <div className="absolute bottom-4 left-4 right-4 bg-black/60 backdrop-blur-sm rounded-lg p-3">
                  <div className="flex items-center justify-between text-white text-sm">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4 text-green-400" />
                      <span>Detecting gesture...</span>
                    </div>
                    <Badge className="bg-teal-500 hover:bg-teal-600">
                      88% confidence
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Output Section */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-gray-700">
                  <MessageSquare className="w-4 h-4" />
                  <span className="text-sm">Text Output</span>
                </div>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-gray-900 text-lg">HELLO</p>
                </div>

                <div className="flex items-center gap-2 text-gray-700">
                  <Volume2 className="w-4 h-4" />
                  <span className="text-sm">Speech Output</span>
                </div>
                <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 flex items-center justify-between">
                  <p className="text-gray-900">"Hello"</p>
                  <Button size="sm" variant="ghost" className="gap-2">
                    <Volume2 className="w-4 h-4" />
                    Play
                  </Button>
                </div>
              </div>

              <Button className="w-full bg-blue-600 hover:bg-blue-700">
                Start Live Translation
              </Button>
            </div>
          </Card>

          {/* Features List */}
          <div className="space-y-6">
            <Card className="p-6 border-l-4 border-l-blue-600">
              <div className="flex gap-4">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center flex-shrink-0">
                  <Video className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <h3 className="text-gray-900 mb-2">Real-Time Recognition</h3>
                  <p className="text-gray-600">
                    Our AI recognizes gestures instantly using your webcam with up to 95% accuracy across multiple sign languages.
                  </p>
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-sm mb-2">
                      <span className="text-gray-600">Processing Speed</span>
                      <span className="text-gray-900">60 FPS</span>
                    </div>
                    <Progress value={95} className="h-2" />
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-6 border-l-4 border-l-teal-600">
              <div className="flex gap-4">
                <div className="w-12 h-12 bg-teal-100 rounded-xl flex items-center justify-center flex-shrink-0">
                  <MessageSquare className="w-6 h-6 text-teal-600" />
                </div>
                <div>
                  <h3 className="text-gray-900 mb-2">Dual Output Format</h3>
                  <p className="text-gray-600">
                    Get instant translation in both text and speech formats, making communication accessible for everyone.
                  </p>
                  <div className="flex gap-2 mt-3">
                    <Badge variant="secondary">Text</Badge>
                    <Badge variant="secondary">Speech</Badge>
                    <Badge variant="secondary">Captions</Badge>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-6 border-l-4 border-l-purple-600">
              <div className="flex gap-4">
                <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center flex-shrink-0">
                  <Activity className="w-6 h-6 text-purple-600" />
                </div>
                <div>
                  <h3 className="text-gray-900 mb-2">Confidence Scoring</h3>
                  <p className="text-gray-600">
                    See real-time confidence scores for each detected gesture to ensure accurate communication.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
}
