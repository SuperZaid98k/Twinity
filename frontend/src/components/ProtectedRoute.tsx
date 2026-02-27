import { Navigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { ReactNode } from "react";

const ProtectedRoute = ({ children }: { children: ReactNode }) => {
    const { isAuthenticated, isLoading } = useAuth();
    if (isLoading) return null;
    if (!isAuthenticated) return <Navigate to="/signin" replace />;
    return <>{children}</>;
};

export default ProtectedRoute;
