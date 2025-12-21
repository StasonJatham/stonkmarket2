import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  InputOTP,
  InputOTPGroup,
  InputOTPSeparator,
  InputOTPSlot,
} from '@/components/ui/input-otp';
import { TrendingUp, AlertCircle, ShieldCheck, ArrowLeft } from 'lucide-react';

export function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [mfaCode, setMfaCode] = useState('');
  const [showMfa, setShowMfa] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  
  const from = location.state?.from?.pathname || '/';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await login(username, password, showMfa ? mfaCode : undefined);
      navigate(from, { replace: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed';
      if (message === 'MFA_REQUIRED') {
        setShowMfa(true);
        setMfaCode('');
      } else {
        setError(message);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleMfaComplete = (value: string) => {
    setMfaCode(value);
    // Auto-submit when 6 digits entered
    if (value.length === 6) {
      handleMfaSubmit(value);
    }
  };

  const handleMfaSubmit = async (code?: string) => {
    setError('');
    setIsLoading(true);
    try {
      await login(username, password, code || mfaCode);
      navigate(from, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Invalid code');
      setMfaCode('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBackToLogin = () => {
    setShowMfa(false);
    setMfaCode('');
    setError('');
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <Card className="w-full max-w-sm">
        <CardHeader className="space-y-1 text-center">
          <div className="flex justify-center mb-2">
            {showMfa ? (
              <ShieldCheck className="h-10 w-10 text-success" />
            ) : (
              <TrendingUp className="h-10 w-10" />
            )}
          </div>
          <CardTitle className="text-2xl font-bold">
            {showMfa ? 'Two-Factor Authentication' : 'StonkMarket'}
          </CardTitle>
          <CardDescription>
            {showMfa 
              ? 'Enter the 6-digit code from your authenticator app'
              : 'Enter your credentials to access admin features'
            }
          </CardDescription>
        </CardHeader>
        <CardContent>
          {error && (
            <div className="flex items-center gap-2 text-sm text-danger p-3 bg-danger/10 rounded-md mb-4">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              {error}
            </div>
          )}

          {showMfa ? (
            <div className="space-y-6">
              <div className="flex justify-center">
                <InputOTP
                  maxLength={6}
                  value={mfaCode}
                  onChange={handleMfaComplete}
                  disabled={isLoading}
                >
                  <InputOTPGroup>
                    <InputOTPSlot index={0} />
                    <InputOTPSlot index={1} />
                    <InputOTPSlot index={2} />
                  </InputOTPGroup>
                  <InputOTPSeparator />
                  <InputOTPGroup>
                    <InputOTPSlot index={3} />
                    <InputOTPSlot index={4} />
                    <InputOTPSlot index={5} />
                  </InputOTPGroup>
                </InputOTP>
              </div>
              
              <Button 
                onClick={() => handleMfaSubmit()} 
                className="w-full" 
                disabled={isLoading || mfaCode.length !== 6}
              >
                {isLoading ? 'Verifying...' : 'Verify Code'}
              </Button>
              
              <Button 
                variant="ghost" 
                onClick={handleBackToLogin}
                className="w-full"
                disabled={isLoading}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Login
              </Button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  type="text"
                  placeholder="admin"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  autoComplete="username"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  autoComplete="current-password"
                />
              </div>
              
              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading ? 'Signing in...' : 'Sign in'}
              </Button>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
