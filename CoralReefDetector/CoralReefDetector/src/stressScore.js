// src/stressScore.js

// Order must match model output:
// ["Bleached_Mild", "Bleached_Moderate", "Bleached_Severe", "Dead", "Healthy"]
const severityLevels = [1, 2, 3, 4, 0];
export const classNames = ["Bleached_Mild", "Bleached_Moderate", "Bleached_Severe", "Dead", "Healthy"];

/**
 * Convert a single prediction (confidence array) to a severity score.
 * @param {number[]} confidences - length 5, softmax output
 * @returns {number} severity score between 0 and 4
 */
export function computeSeverityScore(confidences) {
  if (!Array.isArray(confidences) || confidences.length !== 5) {
    throw new Error('Expected confidence array of length 5');
  }
  let score = 0;
  for (let i = 0; i < 5; i++) {
    score += confidences[i] * severityLevels[i];
  }
  return score;
}

/**
 * Aggregate multiple images for a patch
 * @param {number[][]} predictions - array of confidence arrays
 * @returns {number} average patch stress (0..4)
 */
export function computePatchStress(predictions) {
  if (!Array.isArray(predictions) || predictions.length === 0) {
    throw new Error('No predictions provided');
  }
  let total = 0;
  for (const conf of predictions) {
    total += computeSeverityScore(conf);
  }
  return total / predictions.length;
}

/**
 * Combine patch stress with environment factors into final index.
 * tempFactor and salinityFactor should be normalized to 0..1 if used.
 */
export function computeStressIndex(patchStress, tempFactor = 0, salinityFactor = 0, weights = { patch: 0.5, temp: 0.3, salinity: 0.2 }) {
  return weights.patch * patchStress + weights.temp * tempFactor + weights.salinity * salinityFactor;
}

/**
 * New: Compute dominant label across multiple images
 * @param {Array<number[]>} predictions - array of confidence arrays
 * @returns {Object} { label, confidence, averageScores }
 */
export function computePatchLabel(predictions) {
  if (!Array.isArray(predictions) || predictions.length === 0)
    throw new Error("No predictions provided");

  const numClasses = predictions[0].length;
  const sumScores = new Array(numClasses).fill(0);

  // Sum softmax scores
  predictions.forEach(conf => {
    for (let i = 0; i < numClasses; i++) {
      sumScores[i] += conf[i];
    }
  });

  // Average
  const avgScores = sumScores.map(s => s / predictions.length);

  // Dominant class
  let maxIdx = 0;
  let maxVal = avgScores[0];
  for (let i = 1; i < numClasses; i++) {
    if (avgScores[i] > maxVal) {
      maxVal = avgScores[i];
      maxIdx = i;
    }
  }

  return {
    label: classNames[maxIdx],
    confidence: maxVal,
    averageScores: avgScores
  };
}
