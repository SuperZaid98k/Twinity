import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import ProtectedRoute from "@/components/ProtectedRoute";
import Index from "./pages/Index";
import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import Dashboard from "./pages/Dashboard";
import CreateClone from "./pages/CreateClone";
import Explore from "./pages/Explore";
import ChatClone from "./pages/ChatClone";
import ManageClone from "./pages/ManageClone";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
    <QueryClientProvider client={queryClient}>
        <TooltipProvider>
            <Toaster />
            <Sonner />
            <BrowserRouter>
                <AuthProvider>
                    <Routes>
                        <Route path="/" element={<Index />} />
                        <Route path="/signin" element={<SignIn />} />
                        <Route path="/signup" element={<SignUp />} />
                        <Route path="/explore" element={<Explore />} />
                        <Route path="/chat/:cloneId" element={<ChatClone />} />
                        <Route path="/clone/:cloneId" element={<ProtectedRoute><ManageClone /></ProtectedRoute>} />
                        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
                        <Route path="/create" element={<ProtectedRoute><CreateClone /></ProtectedRoute>} />
                        <Route path="*" element={<NotFound />} />
                    </Routes>
                </AuthProvider>
            </BrowserRouter>
        </TooltipProvider>
    </QueryClientProvider>
);

export default App;
