
// App.tsx
import 'react-native-url-polyfill/auto'
import React, { useState, useEffect } from 'react';
import RNFS from 'react-native-fs'
import { decode } from 'base-64'

global.atob = decode

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
  const [images, setImages] = useState<Array<{ uri: string }>>([]);

  // ✅ NEW ENV INPUTS
  const [SSTA_DHW, setSSTA_DHW] = useState("");
  const [TSA_DHW, setTSA_DHW] = useState("");
  const [SSTA, setSSTA] = useState("");
  const [SSTA_Frequency, setSSTA_Frequency] = useState("");
  const [Temperature_Maximum, setTemperature_Maximum] = useState("");
  const [Turbidity, setTurbidity] = useState("");
  const [Depth_m, setDepth_m] = useState("");

  const [location, setLocation] = useState<{ latitude: number, longitude: number } | null>(null);

  // ✅ OUTPUTS
  const [coralStatus, setCoralStatus] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [bleachingPrediction, setBleachingPrediction] = useState<number | null>(null);

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

      setImages(prev => [...prev, { uri }]);

      setCoralStatus(res.label);
      setConfidence(res.scores ? Math.max(...res.scores) : null);

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
    setCoralStatus(null);
    setBleachingPrediction(null);
    setCollecting(true);
  };

  // ✅ TEMP BLEACHING PREDICTION
  const predictBleaching = async () => {
    try {
      const input = [
        parseFloat(SSTA_DHW),
        parseFloat(TSA_DHW),
        parseFloat(SSTA),
        parseFloat(SSTA_Frequency),
        parseFloat(Temperature_Maximum),
        parseFloat(Turbidity),
        parseFloat(Depth_m),

        // temporal approximations
        parseFloat(TSA_DHW),
        parseFloat(TSA_DHW),
        parseFloat(SSTA),
        parseFloat(TSA_DHW)
      ];

      const response = await fetch("http://192.168.1.34:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ input })
      });

      const data = await response.json();

      if (data.error) {
        Alert.alert("Server Error", data.error);
        return;
      }

      let value = data.prediction;

      // clamp between 0 and 100
      value = Math.max(0, Math.min(100, value));

      setBleachingPrediction(value);

    } catch (err) {
      Alert.alert("Prediction error", String(err));
    }
  };

  const uploadToSupabase = async () => {
    if (!location || images.length === 0) {
      Alert.alert("No data to upload");
      return;
    }

    try {
      const imageUri = images[0].uri;
      const fileName = `coral_${Date.now()}.jpg`;

      const filePath = imageUri.replace('file://', '');
      const base64 = await RNFS.readFile(filePath, 'base64');

      const binaryString = atob(base64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);

      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const { error: uploadError } = await supabase.storage
        .from('coral-images')
        .upload(fileName, bytes, {
          contentType: 'image/jpeg',
          upsert: true,
        });

      if (uploadError) {
        Alert.alert("Image upload failed", JSON.stringify(uploadError));
        return;
      }

      const { data } = supabase.storage
        .from('coral-images')
        .getPublicUrl(fileName);

      const imageUrl = data.publicUrl;

      const payload = {
        image_url: imageUrl,
        latitude: location.latitude,
        longitude: location.longitude,

        coral_status: coralStatus,
        confidence: confidence,

        bleaching_percentage: bleachingPrediction,

        ssta_dhw: parseFloat(SSTA_DHW),
        tsa_dhw: parseFloat(TSA_DHW),
        ssta: parseFloat(SSTA),
        ssta_frequency: parseFloat(SSTA_Frequency),
        temperature_maximum: parseFloat(Temperature_Maximum),
        turbidity: parseFloat(Turbidity),
        depth_m: parseFloat(Depth_m)
      };

      const { error } = await supabase
        .from('coral_scans')
        .insert([payload]);

      if (error) {
        Alert.alert("Database insert failed", error.message);
      } else {
        Alert.alert("Uploaded ✅");
      }

    } catch (err) {
      Alert.alert("Upload error", String(err));
    }
  };

  const clear = () => {
    setImages([]);
    setCoralStatus(null);
    setBleachingPrediction(null);
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

        <Pressable style={styles.button} onPress={predictBleaching}>
          <Text style={styles.buttonText}>Predict Bleaching</Text>
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

      {/* ✅ INPUTS */}
      <View style={{ marginTop: 10 }}>
        <Text>SSTA_DHW</Text>
        <TextInput style={styles.input} value={SSTA_DHW} onChangeText={setSSTA_DHW} />

        <Text>TSA_DHW</Text>
        <TextInput style={styles.input} value={TSA_DHW} onChangeText={setTSA_DHW} />

        <Text>SSTA</Text>
        <TextInput style={styles.input} value={SSTA} onChangeText={setSSTA} />

        <Text>SSTA Frequency</Text>
        <TextInput style={styles.input} value={SSTA_Frequency} onChangeText={setSSTA_Frequency} />

        <Text>Max Temperature</Text>
        <TextInput style={styles.input} value={Temperature_Maximum} onChangeText={setTemperature_Maximum} />

        <Text>Turbidity</Text>
        <TextInput style={styles.input} value={Turbidity} onChangeText={setTurbidity} />

        <Text>Depth (m)</Text>
        <TextInput style={styles.input} value={Depth_m} onChangeText={setDepth_m} />
      </View>

      <ScrollView>
        {images.map((img, i) => (
          <Image key={i} source={{ uri: img.uri }} style={styles.thumbnail} />
        ))}
      </ScrollView>

      {/* ✅ RESULTS */}
      <View style={{ marginTop: 10 }}>
        {coralStatus && <Text>Coral Status: {coralStatus}</Text>}
        {confidence !== null && <Text>Confidence: {confidence.toFixed(2)}</Text>}
        {bleachingPrediction !== null && (
          <Text>Bleaching Prediction: {bleachingPrediction.toFixed(2)}%</Text>
        )}

        {location && (
          <Text>
            GPS: {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
          </Text>
        )}
      </View>
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

