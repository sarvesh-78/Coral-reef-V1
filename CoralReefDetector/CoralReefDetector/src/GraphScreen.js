import React from 'react'
import { View, Text, StyleSheet, ScrollView, Pressable } from 'react-native'
import Svg, { Line, Circle } from 'react-native-svg'

export default function GraphScreen({ siteKey, scans, onBack }) {

  if (!scans || scans.length === 0) {
    return (
      <View style={styles.container}>
        <Pressable style={styles.backBtn} onPress={onBack}>
          <Text style={styles.backText}>← Back</Text>
        </Pressable>
        <Text>No data available</Text>
      </View>
    )
  }

  const graphHeight = 220
  const spacing = 70
  const graphWidth = (scans.length - 1) * spacing + 40

  const values = scans.map(s => Number(s.final_stress_index) || 0)
  const labels = scans.map(s => new Date(s.created_at).toLocaleDateString())

  const normalizeY = (value) => graphHeight - (value * graphHeight)

  const points = values.map((value, index) => ({
    x: index * spacing + 40,
    y: normalizeY(value) + 20
  }))

  return (
    <ScrollView style={styles.container}>

      <Pressable style={styles.backBtn} onPress={onBack}>
        <Text style={styles.backText}>← Back to History</Text>
      </Pressable>

      <Text style={styles.title}>Coral Stress Trend</Text>
      <Text style={styles.subtitle}>📍 {siteKey}</Text>
      <Text style={styles.yAxisLabel}>Final Stress Index (0 - 1)</Text>

      <ScrollView horizontal>

        <View style={{ paddingBottom: 60 }}>

          <Svg height={graphHeight + 60} width={graphWidth + 40}>

            {/* X AXIS */}
            <Line
              x1="40"
              y1={graphHeight + 20}
              x2={graphWidth + 40}
              y2={graphHeight + 20}
              stroke="black"
              strokeWidth="2"
            />

            {/* Y AXIS */}
            <Line
              x1="40"
              y1="20"
              x2="40"
              y2={graphHeight + 20}
              stroke="black"
              strokeWidth="2"
            />

            {/* Y SCALE */}
            {[1, 0.75, 0.5, 0.25, 0].map((val, i) => {
              const y = normalizeY(val) + 20
              return (
                <React.Fragment key={i}>
                  <Line
                    x1="35"
                    y1={y}
                    x2="40"
                    y2={y}
                    stroke="black"
                    strokeWidth="2"
                  />
                </React.Fragment>
              )
            })}

            {/* CONNECTING LINES */}
            {points.map((point, i) => {
              if (i === 0) return null
              const prev = points[i - 1]
              return (
                <Line
                  key={i}
                  x1={prev.x}
                  y1={prev.y}
                  x2={point.x}
                  y2={point.y}
                  stroke="#ff3b30"
                  strokeWidth="2"
                />
              )
            })}

            {/* POINTS */}
            {points.map((point, i) => (
              <Circle
                key={i}
                cx={point.x}
                cy={point.y}
                r="4"
                fill="#007AFF"
              />
            ))}

          </Svg>

          {/* VALUES */}
          {points.map((point, i) => (
            <Text
              key={i}
              style={{
                position: 'absolute',
                left: point.x - 15,
                top: point.y - 20,
                fontSize: 10
              }}
            >
              {values[i].toFixed(2)}
            </Text>
          ))}

          {/* DATES */}
          {points.map((point, i) => (
            <Text
              key={i}
              style={{
                position: 'absolute',
                left: point.x - 35,
                top: graphHeight + 25,
                width: 70,
                fontSize: 10,
                textAlign: 'center'
              }}
            >
              {labels[i]}
            </Text>
          ))}

        </View>
      </ScrollView>

      <Text style={styles.xAxisLabel}>Time (Date)</Text>

    </ScrollView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 12,
    backgroundColor: '#fff'
  },
  backBtn: {
    backgroundColor: '#007AFF',
    padding: 10,
    borderRadius: 8,
    marginBottom: 12,
    alignItems: 'center'
  },
  backText: {
    color: '#fff',
    fontWeight: 'bold'
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold'
  },
  subtitle: {
    color: 'gray',
    marginBottom: 6
  },
  yAxisLabel: {
    fontSize: 12,
    fontWeight: 'bold',
    marginBottom: 6
  },
  xAxisLabel: {
    marginTop: 25,
    textAlign: 'center',
    fontWeight: 'bold'
  }
})
