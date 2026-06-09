import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ConnectionState } from '../hooks/useWebSocket';

interface ConnectionStatusProps {
  status: ConnectionState;
}

export function ConnectionStatus({ status }: ConnectionStatusProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return '#10B981'; // Green
      case 'connecting':
      case 'reconnecting':
        return '#F59E0B'; // Yellow/Orange
      case 'disconnected':
      default:
        return '#EF4444'; // Red
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connected':
        return 'Live';
      case 'connecting':
        return 'Connecting...';
      case 'reconnecting':
        return 'Reconnecting...';
      case 'disconnected':
      default:
        return 'Offline';
    }
  };

  return (
    <View style={styles.container}>
      <View style={[styles.dot, { backgroundColor: getStatusColor() }]} />
      <Text style={styles.text}>{getStatusText()}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1F2937',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  text: {
    color: '#F3F4F6',
    fontSize: 12,
    fontWeight: '600',
  },
});
