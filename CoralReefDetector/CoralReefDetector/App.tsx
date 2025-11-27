// App.tsx
import 'react-native-url-polyfill/auto'
import React, { useState, useEffect } from 'react';
import {
  StatusBar,
  StyleSheet,
  View,
  Text,
  Image,
  PermissionsAndroid,
  Platform,
  Alert,
  ScrollView,
  Pressable,
  TextInput
} from 'react-native';

import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import CoralDetector from './src/CoralDetector';
import { launchImageLibrary, launchCamera } from 'react-native-image-picker';
import Geolocation from 'react-native-geolocation-service';

import {
  computePatchLabel,
  computePatchStress,
  computeStressIndex
} from './src/stressScore';

import { supabase } from './src/supabaseClient';
import HistoryScreen from './src/historyScreen';

function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />
      <AppContent />
    </SafeAreaProvider>
  );
}

function AppContent() {
  const insets = useSafeAreaInsets();

  const [showHistory, setShowHistory] = useState(false);

  const [modelLoaded, setModelLoaded] = useState(false);
  const [collecting, setCollecting] = useState(false);
  const [images, setImages] = useState<Array<{ uri: string; scores?: number[] }>>([]);

  const [surfaceTemp, setSurfaceTemp] = useState("28");
  const [wqi, setWqi] = useState("80");
  const [ph, setPh] = useState("8.1");

  const [location, setLocation] = useState<{ latitude: number, longitude: number } | null>(null);
  const [patchResult, setPatchResult] = useState<any>(null);

  useEffect(() => {
    (async () => {
      try {
        await CoralDetector.loadModel();
        setModelLoaded(true);
      } catch (e) {
        console.warn('Model load failed:', e);
      }
    })();
  }, []);

  const requestPermissions = async () => {
    if (Platform.OS !== 'android') return true;

    const perms = [
      PermissionsAndroid.PERMISSIONS.CAMERA,
      PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION
    ];

    if (Platform.Version >= 33) perms.push('android.permission.READ_MEDIA_IMAGES');
    else perms.push(PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE);

    const granted = await PermissionsAndroid.requestMultiple(perms);
    return Object.values(granted).every(v => v === PermissionsAndroid.RESULTS.GRANTED);
  };

  const requestLocation = async () => {
    return new Promise<{ latitude: number, longitude: number }>((resolve, reject) => {
      Geolocation.getCurrentPosition(
        position => {
          const { latitude, longitude } = position.coords;
          resolve({ latitude, longitude });
        },
        error => reject(error),
        { enableHighAccuracy: true, timeout: 15000 }
      );
    });
  };

  const handleSingleImage = async (uri: string) => {
    try {
      const filePath = await CoralDetector.copyContentUriToFile(uri);
      const res = await CoralDetector.predictCoral(filePath);
      const scores = Array.isArray(res.scores) ? res.scores : null;

      setImages(prev => [...prev, { uri, scores: scores ?? undefined }]);
    } catch (e) {
      Alert.alert('Prediction failed', String(e));
    }
  };

  const pickImage = async () => {
    const ok = await requestPermissions();
    if (!ok) return Alert.alert('Permissions required');

    launchImageLibrary({ mediaType: 'photo' }, async (response) => {
      if (!response.assets?.[0]?.uri) return;
      if (!collecting) return Alert.alert('Start patch first');
      await handleSingleImage(response.assets[0].uri);
    });
  };

  const takePhoto = async () => {
    const ok = await requestPermissions();
    if (!ok) return;

    launchCamera({ mediaType: 'photo' }, async (response) => {
      if (!response.assets?.[0]?.uri) return;
      if (!collecting) return Alert.alert('Start patch first');
      await handleSingleImage(response.assets[0].uri);
    });
  };

  const startPatch = async () => {
    const loc = await requestLocation();
    setLocation(loc);
    setImages([]);
    setPatchResult(null);
    setCollecting(true);
  };

  const calculateTempStress = (temp: number) => {
    if (temp <= 29) return 0.2;
    if (temp <= 31) return 0.6;
    return 1.0;
  };

  const calculatePollutionStress = (wqi: number) => {
    if (wqi >= 80) return 0.2;
    if (wqi >= 60) return 0.6;
    return 1.0;
  };

  const calculateAcidStress = (ph: number) => {
    if (ph >= 8 && ph <= 8.4) return 0.2;
    if (ph >= 7.7) return 0.6;
    return 1.0;
  };

  const getStressLabel = (value: number) => {
    if (value === 1.0) return "HIGH";
    if (value === 0.6) return "MODERATE";
    return "LOW";
  };

  const finishPatch = () => {
    const validScores = images.map(i => i.scores).filter(Boolean) as number[][];
    if (validScores.length === 0 || !location) {
      Alert.alert('Missing data');
      return;
    }

    const temperature = parseFloat(surfaceTemp);
    const waterQuality = parseFloat(wqi);
    const phLevel = parseFloat(ph);

    const result = computePatchLabel(validScores);
    const patchStress = computePatchStress(validScores);
    const normalizedStress = patchStress / 4;

    const tempStress = calculateTempStress(temperature);
    const pollStress = calculatePollutionStress(waterQuality);
    const acidStress = calculateAcidStress(phLevel);

    const all = [
      { name: 'Temperature', value: tempStress },
      { name: 'Pollution', value: pollStress },
      { name: 'Acidification', value: acidStress }
    ];

    const mainStress = all.sort((a, b) => b.value - a.value)[0].name;

    const finalStressIndex = computeStressIndex(
      normalizedStress,
      tempStress,
      pollStress
    );

    let recovery = 'High';
    if (result.label === "Dead") {
      recovery = 'Low'
    }
    else if (finalStressIndex > 0.7) {
      recovery = 'Low'
    }
    else if (finalStressIndex > 0.4) {
      recovery = 'Moderate'
    }

    setPatchResult({
      ...result,
      patchStress,
      finalStressIndex,
      recovery,
      mainStress,
      tempStress,
      pollutionStress: pollStress,
      acidStress
    });

    setCollecting(false);
  };

  const uploadToSupabase = async () => {
    if (!patchResult || !location) {
      Alert.alert("No data to upload");
      return;
    }

    const payload = {
      image_url: images[0]?.uri,
      latitude: location.latitude,
      longitude: location.longitude,
      surface_temp: parseFloat(surfaceTemp),
      wqi: parseFloat(wqi),
      ph: parseFloat(ph),
      patch_stress: patchResult.patchStress,
      final_stress_index: patchResult.finalStressIndex,
      main_stress_factor: patchResult.mainStress,
      dominant_label: patchResult.label,
      recovery: patchResult.recovery
    };

    const { error } = await supabase.from('coral_scans').insert([payload]);

    if (error) Alert.alert("Upload Failed", error.message);
    else Alert.alert("Uploaded ✅");
  };

  const clear = () => {
    setImages([]);
    setPatchResult(null);
    setCollecting(false);
  };

  if (showHistory) {
    return (
      <View style={{ flex: 1 }}>
        <HistoryScreen onBack={() => setShowHistory(false)} />
      </View>
    );
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.buttonsRow}>
        <Pressable style={styles.button} onPress={startPatch}>
          <Text style={styles.buttonText}>{collecting ? "Collecting..." : "Start Patch"}</Text>
        </Pressable>

        <Pressable style={styles.button} onPress={pickImage}>
          <Text style={styles.buttonText}>Pick Image</Text>
        </Pressable>

        <Pressable style={styles.button} onPress={takePhoto}>
          <Text style={styles.buttonText}>Camera</Text>
        </Pressable>

        <Pressable style={styles.button} onPress={finishPatch}>
          <Text style={styles.buttonText}>Finish</Text>
        </Pressable>

        <Pressable style={styles.button} onPress={uploadToSupabase}>
          <Text style={styles.buttonText}>Upload</Text>
        </Pressable>

        <Pressable style={styles.button} onPress={() => setShowHistory(true)}>
          <Text style={styles.buttonText}>History</Text>
        </Pressable>

        <Pressable style={styles.button} onPress={clear}>
          <Text style={styles.buttonText}>Clear</Text>
        </Pressable>
      </View>

      <View style={{ marginTop: 10 }}>
        <Text>Temperature (°C)</Text>
        <TextInput style={styles.input} value={surfaceTemp} onChangeText={setSurfaceTemp} />

        <Text>WQI</Text>
        <TextInput style={styles.input} value={wqi} onChangeText={setWqi} />

        <Text>pH</Text>
        <TextInput style={styles.input} value={ph} onChangeText={setPh} />
      </View>

      <ScrollView>
        {images.map((img, i) => (
          <Image key={i} source={{ uri: img.uri }} style={styles.thumbnail} />
        ))}
      </ScrollView>

      {patchResult && (
        <View style={{ marginTop: 10 }}>
          <Text>Dominant Label: {patchResult.label}</Text>
          <Text>Main Stress: {patchResult.mainStress}</Text>
          <Text>FSI: {patchResult.finalStressIndex.toFixed(2)}</Text>
          <Text>Recovery: {patchResult.recovery}</Text>

          <Text>Temperature: {getStressLabel(patchResult.tempStress)}</Text>
          <Text>Pollution: {getStressLabel(patchResult.pollutionStress)}</Text>
          <Text>Acidification: {getStressLabel(patchResult.acidStress)}</Text>

          {location && (
            <Text>
              GPS: {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
            </Text>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 12, backgroundColor: '#fff' },
  buttonsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  button: { backgroundColor: '#007AFF', padding: 10, borderRadius: 6 },
  buttonText: { color: '#fff', fontWeight: 'bold' },
  input: { borderWidth: 1, borderColor: '#aaa', marginVertical: 4, padding: 6 },
  thumbnail: { width: 120, height: 120, marginTop: 8 }
});

export default App;
