import React, { useEffect, useState } from 'react'
import {
  View,
  Text,
  ScrollView,
  Image,
  StyleSheet,
  ActivityIndicator,
  Pressable
} from 'react-native'

import { supabase } from './supabaseClient'
import GraphScreen from './GraphScreen'

export default function HistoryScreen({ onBack }) {

  const [loading, setLoading] = useState(true)
  const [grouped, setGrouped] = useState({})
  const [selectedSite, setSelectedSite] = useState(null)

  useEffect(() => {
    fetchScans()
  }, [])

  const fetchScans = async () => {
    const { data, error } = await supabase
      .from('coral_scans')
      .select('*')
      .order('created_at', { ascending: true })

    if (error) {
      console.error(error)
      setLoading(false)
      return
    }

    const groupedData = {}

    data.forEach(scan => {
      if (scan.latitude == null || scan.longitude == null) return

      const key = `${scan.latitude}, ${scan.longitude}`

      if (!groupedData[key]) {
        groupedData[key] = {
          location: key,
          scans: []
        }
      }

      groupedData[key].scans.push(scan)
    })

    setGrouped(groupedData)
    setLoading(false)
  }

  // ---------- GRAPH VIEW ----------
  if (selectedSite) {
    return (
      <GraphScreen
        siteKey={selectedSite.location}
        scans={selectedSite.scans}
        onBack={() => setSelectedSite(null)}
      />
    )
  }

  if (loading) {
    return <ActivityIndicator style={{ marginTop: 50 }} size="large" />
  }

  const keys = Object.keys(grouped)

  return (
    <ScrollView style={styles.container}>

      <Pressable style={styles.backBtn} onPress={onBack}>
        <Text style={styles.backText}>← Back to Main</Text>
      </Pressable>

      <Text style={styles.title}>Coral Site History</Text>

      {keys.map((key, index) => {
        const group = grouped[key]

        return (
          <View key={index} style={styles.locationBlock}>

            <Text style={styles.locationTitle}>📍 {group.location}</Text>

            <Pressable
              style={styles.graphBtn}
              onPress={() => setSelectedSite(group)}
            >
              <Text style={{ color: '#fff', fontWeight: 'bold' }}>
                View Graph
              </Text>
            </Pressable>

            {group.scans.map((scan, i) => (
              <View key={i} style={styles.card}>

                {scan.image_url && (
                  <Image
                    source={{ uri: scan.image_url }}
                    style={styles.image}
                  />
                )}

                <Text>Label: {scan.dominant_label}</Text>
                <Text>Main Stress: {scan.main_stress_factor}</Text>
                <Text>
                  Final Stress Index: {Number(scan.final_stress_index).toFixed(2)}
                </Text>
                <Text>Recovery: {scan.recovery}</Text>
                <Text>Temp: {scan.surface_temp}</Text>
                <Text>WQI: {scan.wqi}</Text>
                <Text>pH: {scan.ph}</Text>

                <Text style={styles.date}>
                  {new Date(scan.created_at).toLocaleString()}
                </Text>

              </View>
            ))}
          </View>
        )
      })}

      {keys.length === 0 && (
        <Text style={{ textAlign: 'center', marginTop: 20 }}>
          No data available yet
        </Text>
      )}

    </ScrollView>
  )
}

const styles = StyleSheet.create({
  container: {
    padding: 12,
    backgroundColor: '#fff',
    flex: 1
  },
  backBtn: {
    backgroundColor: '#007AFF',
    padding: 10,
    borderRadius: 8,
    marginBottom: 10,
    alignItems: 'center'
  },
  backText: {
    color: '#fff',
    fontWeight: 'bold'
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10
  },
  locationBlock: {
    marginBottom: 30,
    borderBottomWidth: 2,
    borderColor: '#0a84ff',
    paddingBottom: 10
  },
  locationTitle: {
    fontSize: 15,
    fontWeight: 'bold'
  },
  graphBtn: {
    backgroundColor: '#34C759',
    padding: 8,
    borderRadius: 6,
    marginVertical: 6,
    alignItems: 'center'
  },
  card: {
    padding: 12,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 10,
    marginBottom: 12,
    marginTop: 5
  },
  image: {
    width: '100%',
    height: 180,
    borderRadius: 8,
    marginBottom: 8
  },
  date: {
    marginTop: 6,
    fontSize: 12,
    color: 'gray'
  }
})
