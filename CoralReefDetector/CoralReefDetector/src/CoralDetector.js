// src/CoralDetector.js
import { NativeModules, Platform } from 'react-native';
const { CoralModule, CoralDetector } = NativeModules; // try both keys; RN autolinking may export different names

// helper to get the available native object
const Native = CoralModule || CoralDetector || NativeModules.CoralModule || NativeModules.CoralDetector;

/**
 * Copies a content URI to a temporary file (Android only).
 * On iOS, the image picker often provides a file path already.
 */
export async function copyContentUriToFile(uri) {
  if (Platform.OS === 'android') {
    if (!Native || !Native.copyContentUriToFile) throw new Error('Native method copyContentUriToFile not available');
    return await Native.copyContentUriToFile(uri);
  }
  return uri;
}

/**
 * Calls native model loader if present (optional).
 */
export async function loadModel() {
  if (!Native || !Native.loadModel) return null;
  return await Native.loadModel();
}

/**
 * Predict: expects native method to return either:
 * - a JSON string: '{"label":"Healthy","scores":[...]}'
 * - OR a JS object/map: { label: 'Healthy', scores: [ ... ] }
 *
 * This function normalizes either return type to { label, scores }.
 */
export async function predictCoral(filePath) {
  if (!Native) throw new Error('Native Coral module not available');
  // prefer predict() if available, otherwise classifyImage()
  const fn = Native.predict ? Native.predict : (Native.classifyImage ? Native.classifyImage : null);
  if (!fn) throw new Error('Native prediction method not found (predict/classifyImage)');
  const raw = await fn(filePath);
  // raw might be a JSON string (older wrappers) or an object/map
  if (typeof raw === 'string') {
    try {
      const parsed = JSON.parse(raw);
      return parsed; // expects { label, scores }
    } catch (e) {
      throw new Error('Native returned string but JSON parse failed: ' + e.message);
    }
  }
  // If raw is a Native map, it may come as an object with label and scores
  return raw;
}

export default {
  copyContentUriToFile,
  loadModel,
  predictCoral,
};
