import React, { memo } from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import { ViolationAlert, ViolationSeverity } from '../types';

interface ViolationAlertCardProps {
  alert: ViolationAlert;
}

const getSeverityColor = (severity: ViolationSeverity) => {
  switch (severity) {
    case 'CRITICAL':
      return '#EF4444'; // Red
    case 'HIGH':
      return '#F59E0B'; // Orange
    case 'MEDIUM':
      return '#FCD34D'; // Yellow
    default:
      return '#6B7280'; // Gray
  }
};

const formatTimeAgo = (dateStr: string) => {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  
  if (diffMs < 60000) return `${Math.floor(diffMs / 1000)}s ago`;
  if (diffMs < 3600000) return `${Math.floor(diffMs / 60000)}m ago`;
  return `${Math.floor(diffMs / 3600000)}h ago`;
};

// Memoized to prevent re-renders when parent list updates
export const ViolationAlertCard = memo(({ alert }: ViolationAlertCardProps) => {
  const severityColor = getSeverityColor(alert.severity);

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.trackId}>Track #{alert.track_id}</Text>
          <Text style={styles.timestamp}>{formatTimeAgo(alert.timestamp_utc)}</Text>
        </View>
        <View style={[styles.badge, { backgroundColor: severityColor }]}>
          <Text style={styles.badgeText}>{alert.severity}</Text>
        </View>
      </View>

      <View style={styles.content}>
        {alert.thumbnail_b64 ? (
          <Image
            source={{ uri: `data:image/jpeg;base64,${alert.thumbnail_b64}` }}
            style={styles.thumbnail}
            resizeMode="cover"
          />
        ) : (
          <View style={styles.thumbnailPlaceholder}>
            <Text style={styles.placeholderText}>No Image</Text>
          </View>
        )}

        <View style={styles.details}>
          {alert.license_plate && (
            <Text style={styles.licensePlate}>{alert.license_plate}</Text>
          )}

          <View style={styles.violationsContainer}>
            {alert.violations.map((v, index) => (
              <View key={`${v.type}-${index}`} style={styles.violationChip}>
                <Text style={styles.violationText}>
                  {v.type} ({Math.round(v.confidence * 100)}%)
                </Text>
              </View>
            ))}
          </View>
          
          <Text style={styles.actionText}>{alert.recommended_action.replace(/_/g, ' ')}</Text>
        </View>
      </View>
    </View>
  );
});

// Provide a custom comparator if needed, but shallow comparison of `alert` object is usually sufficient
// if it's treated immutably by the parent.

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  trackId: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
    marginRight: 8,
  },
  timestamp: {
    fontSize: 12,
    color: '#6B7280',
  },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  badgeText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  content: {
    flexDirection: 'row',
  },
  thumbnail: {
    width: 80,
    height: 80,
    borderRadius: 8,
    marginRight: 12,
    backgroundColor: '#E5E7EB',
  },
  thumbnailPlaceholder: {
    width: 80,
    height: 80,
    borderRadius: 8,
    marginRight: 12,
    backgroundColor: '#F3F4F6',
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    fontSize: 10,
    color: '#9CA3AF',
  },
  details: {
    flex: 1,
    justifyContent: 'space-between',
  },
  licensePlate: {
    fontFamily: 'monospace',
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    backgroundColor: '#F3F4F6',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    alignSelf: 'flex-start',
    marginBottom: 6,
  },
  violationsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
  },
  violationChip: {
    backgroundColor: '#EFF6FF',
    borderWidth: 1,
    borderColor: '#BFDBFE',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 6,
  },
  violationText: {
    fontSize: 11,
    color: '#1D4ED8',
    fontWeight: '500',
  },
  actionText: {
    fontSize: 12,
    color: '#4B5563',
    fontStyle: 'italic',
    marginTop: 4,
  },
});
