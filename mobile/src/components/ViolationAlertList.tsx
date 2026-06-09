import React, { useCallback } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { FlashList } from '@shopify/flash-list';
import { ViolationAlert } from '../types';
import { ViolationAlertCard } from './ViolationAlertCard';

interface ViolationAlertListProps {
  alerts: ViolationAlert[];
}

export function ViolationAlertList({ alerts }: ViolationAlertListProps) {
  // Use a composite key because track_id alone isn't unique over time,
  // and we might receive multiple events for the same track at different frames.
  const keyExtractor = useCallback(
    (item: ViolationAlert) => `${item.track_id}-${item.frame_id}-${item.timestamp_utc}`,
    []
  );

  const renderItem = useCallback(({ item }: { item: ViolationAlert }) => {
    return <ViolationAlertCard alert={item} />;
  }, []);

  if (alerts.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyText}>No alerts yet. Monitoring field...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {React.createElement(FlashList as any, {
        data: alerts,
        renderItem: renderItem,
        keyExtractor: keyExtractor,
        estimatedItemSize: 120,
        contentContainerStyle: styles.listContent,
        showsVerticalScrollIndicator: false,
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    width: '100%',
  },
  listContent: {
    paddingVertical: 16,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    color: '#6B7280',
    fontSize: 16,
    fontStyle: 'italic',
  },
});
