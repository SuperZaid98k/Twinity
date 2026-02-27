import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { toast } from "sonner";
import { getCurrentUser, signIn as apiSignIn, signOut as apiSignOut, signUp as apiSignUp } from "@/lib/api";

interface AuthContextType {
    isAuthenticated: boolean;
    isLoading: boolean;
    user: { name: string; username: string } | null;
    signIn: (username: string, password: string) => Promise<boolean>;
    signUp: (name: string, username: string, password: string) => Promise<boolean>;
    signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth must be used within AuthProvider");
    return ctx;
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<{ name: string; username: string } | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        let mounted = true;

        const loadSession = async () => {
            try {
                const response = await getCurrentUser();
                if (mounted && response?.user) {
                    setUser(response.user);
                }
            } catch {
                if (mounted) {
                    setUser(null);
                }
            } finally {
                if (mounted) {
                    setIsLoading(false);
                }
            }
        };

        loadSession();
        return () => {
            mounted = false;
        };
    }, []);

    const signIn = async (username: string, password: string) => {
        if (!username || !password) {
            toast.error("Please fill in all fields");
            return false;
        }

        try {
            const response = await apiSignIn(username, password);
            setUser(response.user);
            toast.success("Signed in successfully!");
            return true;
        } catch (error) {
            const message = error instanceof Error ? error.message : "Sign in failed";
            toast.error(message);
            return false;
        }
    };

    const signUp = async (name: string, username: string, password: string) => {
        if (!name || !username || !password) {
            toast.error("Please fill in all fields");
            return false;
        }

        try {
            const response = await apiSignUp(name, username, password);
            setUser(response.user);
            toast.success("Account created successfully!");
            return true;
        } catch (error) {
            const message = error instanceof Error ? error.message : "Sign up failed";
            toast.error(message);
            return false;
        }
    };

    const signOut = async () => {
        try {
            await apiSignOut();
        } catch {
            // Ignore backend errors here; local session should still clear.
        } finally {
            setUser(null);
            toast.success("Signed out");
        }
    };

    return (
        <AuthContext.Provider value={{ isAuthenticated: !!user, isLoading, user, signIn, signUp, signOut }}>
            {children}
        </AuthContext.Provider>
    );
};
