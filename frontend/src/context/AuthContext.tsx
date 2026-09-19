"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from "react";
import { User, registerUser, loginUser, getCurrentUser } from "@/lib/api";

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const TOKEN_KEY = "citerag_access_token";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Initialize session from localStorage
  useEffect(() => {
    let active = true;
    const initAuth = async () => {
      try {
        const storedToken = localStorage.getItem(TOKEN_KEY);
        if (storedToken) {
          const profile = await getCurrentUser(storedToken);
          if (active) {
            setToken(storedToken);
            setUser(profile);
          }
        }
      } catch (err) {
        console.warn("Stored auth token invalid or expired:", err);
        localStorage.removeItem(TOKEN_KEY);
      } finally {
        if (active) {
          setIsLoading(false);
        }
      }
    };

    initAuth();
    return () => {
      active = false;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const authData = await loginUser(email, password);
    localStorage.setItem(TOKEN_KEY, authData.access_token);
    setToken(authData.access_token);
    setUser(authData.user);
  }, []);

  const signup = useCallback(
    async (email: string, password: string, fullName?: string) => {
      const authData = await registerUser(email, password, fullName);
      localStorage.setItem(TOKEN_KEY, authData.access_token);
      setToken(authData.access_token);
      setUser(authData.user);
    },
    []
  );

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user && !!token,
        isLoading,
        login,
        signup,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
