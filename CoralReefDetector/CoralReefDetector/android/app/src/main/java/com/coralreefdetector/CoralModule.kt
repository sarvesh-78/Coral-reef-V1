package com.coralreefdetector

import android.graphics.Bitmap
import com.facebook.react.bridge.*
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter

class CoralModule(reactContext: ReactApplicationContext) :
        ReactContextBaseJavaModule(reactContext) {

    private var interpreter: Interpreter? = null
    private val inputSize = 224
    private val numClasses = 5
    private val classNames =
            arrayOf("Bleached_Mild", "Bleached_Moderate", "Bleached_Severe", "Dead", "Healthy")

    override fun getName(): String = "CoralModule"

    init {
        try {
            loadModel()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /** Copy content URI to temp file for Android */
    @ReactMethod
    fun copyContentUriToFile(uriString: String, promise: Promise) {
        try {
            val uri = android.net.Uri.parse(uriString)
            val inputStream: InputStream? =
                    reactApplicationContext.contentResolver.openInputStream(uri)
            val tempFile = File.createTempFile("coral_", ".jpg", reactApplicationContext.cacheDir)
            FileOutputStream(tempFile).use { output -> inputStream?.copyTo(output) }
            inputStream?.close()
            promise.resolve(tempFile.absolutePath)
        } catch (e: Exception) {
            promise.reject("COPY_FAILED", e)
        }
    }

    /** Predict (classify image and return full scores) */
    @ReactMethod
    fun classifyImage(imagePath: String, promise: Promise) {
        try {
            if (interpreter == null) {
                promise.reject("MODEL_NOT_LOADED", "TFLite model not loaded")
                return
            }

            val bitmap: Bitmap? = ImageUtils.loadBitmapFromPath(imagePath)
            if (bitmap == null) {
                promise.reject("INVALID_IMAGE", "Failed to load image")
                return
            }

            val inputBuffer = ImageUtils.bitmapToFloatBuffer(bitmap, inputSize, inputSize)
            val outputArray = Array(1) { FloatArray(numClasses) }

            interpreter!!.run(inputBuffer, outputArray)

            // Find max class
            var maxIdx = 0
            var maxVal = outputArray[0][0]
            for (i in 1 until numClasses) {
                if (outputArray[0][i] > maxVal) {
                    maxVal = outputArray[0][i]
                    maxIdx = i
                }
            }

            val result = Arguments.createMap()
            result.putString("label", classNames[maxIdx])

            // Add scores array
            val scoresArray = Arguments.createArray()
            for (i in 0 until numClasses) {
                scoresArray.pushDouble(outputArray[0][i].toDouble())
            }
            result.putArray("scores", scoresArray)

            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("PREDICT_FAILED", e)
        }
    }

    /** Optional method to call model load explicitly from JS */
    @ReactMethod
    fun loadModelPromise(promise: Promise) {
        try {
            loadModel()
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("MODEL_LOAD_FAILED", e)
        }
    }

    /** Load TFLite model from assets */
    private fun loadModel() {
        val afd = reactApplicationContext.assets.openFd("coral_model_finetuned.tflite")
        val fileChannel = afd.createInputStream().channel
        val mapped: MappedByteBuffer =
                fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        interpreter = Interpreter(mapped)
    }
}
