import React, { useState, useCallback, useMemo } from 'react';
import { View, Text, StyleSheet, SafeAreaView, ActivityIndicator } from 'react-native';
import { useSilentAuth } from '../hooks/useSilentAuth';
import { useWebSocket } from '../hooks/useWebSocket';
import { ViolationAlert } from '../types';
import { ConnectionStatus } from '../components/ConnectionStatus';
import { ViolationAlertList } from '../components/ViolationAlertList';

// Backend WebSocket and API URLs (configurable via environment in a real app)
// 10.0.2.2 is the localhost alias for Android emulators
const API_BASE_URL = 'http://10.0.2.2:8000';
const WS_URL = 'ws://10.0.2.2:8000/ws/alerts';

const MAX_ALERTS = 200;

export function DashboardScreen() {
  const [alerts, setAlerts] = useState<ViolationAlert[]>([]);
  
  // 1. Silent Authentication (Number Verification)
  const { status: authStatus, error: authError } = useSilentAuth(API_BASE_URL);

  // 2. Real-time WebSocket connection (only enabled if authenticated)
  const isAuth = authStatus === 'verified';
  
  const handleNewAlert = useCallback((alert: ViolationAlert) => {
    setAlerts((prevAlerts) => {
      // Prepend the new alert, and cap the list to avoid OOM crashes on mobile
      const updated = [alert, ...prevAlerts];
      if (updated.length > MAX_ALERTS) {
        updated.pop();
      }
      return updated;
    });
  }, []);

  const { connectionState } = useWebSocket({
    url: WS_URL,
    onMessage: handleNewAlert,
    enabled: isAuth,
  });

  const criticalCount = useMemo(() => alerts.filter(a => a.severity === 'CRITICAL').length, [alerts]);

  // ── Rendering states ────────────────────────────────────

  if (authStatus === 'verifying') {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#3B82F6" />
        <Text style={styles.statusText}>5G Silent Authentication...</Text>
      </View>
    );
  }

  if (authStatus === 'failed') {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>Authentication Failed</Text>
        <Text style={styles.statusText}>{authError}</Text>
      </View>
    );
  }

  // ── Main Dashboard ──────────────────────────────────────
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>MAS Dashboard</Text>
          <Text style={styles.subtitle}>Field Operator View</Text>
        </View>
        <ConnectionStatus status={connectionState} />
      </View>

      <View style={styles.statsBar}>
        <View style={styles.statBox}>
          <Text style={styles.statValue}>{alerts.length}</Text>
          <Text style={styles.statLabel}>Total Alerts</Text>
        </View>
        <View style={styles.statBox}>
          <Text style={[styles.statValue, { color: '#EF4444' }]}>{criticalCount}</Text>
          <Text style={styles.statLabel}>Critical</Text>
        </View>
      </View>

      <ViolationAlertList alerts={alerts} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  title: {
    fontSize: 20,
    fontWeight: '800',
    color: '#111827',
  },
  subtitle: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 2,
  },
  statsBar: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#111827',
  },
  statLabel: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 4,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  statusText: {
    marginTop: 16,
    color: '#4B5563',
    fontSize: 16,
    textAlign: 'center',
  },
  errorText: {
    color: '#EF4444',
    fontSize: 18,
    fontWeight: '700',
  },
});
