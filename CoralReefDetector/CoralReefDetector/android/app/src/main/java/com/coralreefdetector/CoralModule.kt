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

    override fun getName(): String = "CoralModule"

    init {
        try {
            loadModel()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

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

            // Single output (sigmoid)
            val outputArray = Array(1) { FloatArray(1) }

            interpreter!!.run(inputBuffer, outputArray)

            val prob = outputArray[0][0]

            val threshold = 0.6f
            val label = if (prob > threshold) "Healthy" else "Bleached"

            val result = Arguments.createMap()
            result.putString("label", label)

            val scoresArray = Arguments.createArray()
            scoresArray.pushDouble(prob.toDouble())
            result.putArray("scores", scoresArray)

            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("PREDICT_FAILED", e)
        }
    }

    @ReactMethod
    fun loadModelPromise(promise: Promise) {
        try {
            loadModel()
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("MODEL_LOAD_FAILED", e)
        }
    }

    private fun loadModel() {
        val afd = reactApplicationContext.assets.openFd("mobilenet_binary_model.tflite")
        val fileChannel = afd.createInputStream().channel
        val mapped: MappedByteBuffer =
                fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        interpreter = Interpreter(mapped)
    }
}
