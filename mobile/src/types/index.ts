/**
 * Type definitions for the MAS Mobile Dashboard.
 * 
 * These interfaces precisely mirror the Python Pydantic models
 * emitted by the Decision Agent.
 */

export type ViolationSeverity = 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface DetectedViolation {
  type: string;
  confidence: number;
}

export interface ViolationAlert {
  track_id: number;
  frame_id: number;
  license_plate: string | null;
  violations: DetectedViolation[];
  severity: ViolationSeverity;
  qod_session_id: string | null;
  qod_status: string | null;
  recommended_action: string;
  timestamp_utc: string;
  thumbnail_b64?: string; // Optional base64 cropped frame
}
