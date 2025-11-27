package com.coralreefdetector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImageUtils {

    /** Load a bitmap from file path */
    fun loadBitmapFromPath(path: String): Bitmap? {
        return try {
            val file = File(path)
            if (!file.exists()) return null
            BitmapFactory.decodeStream(FileInputStream(file))
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * Convert bitmap to Float32 ByteBuffer normalized to [0,1] RGB order, matches Python
     * preprocessing.
     */
    fun bitmapToFloatBuffer(bitmap: Bitmap, width: Int = 224, height: Int = 224): ByteBuffer {
        // Resize exactly to width x height
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)

        // Allocate buffer: float32 = 4 bytes
        val buffer = ByteBuffer.allocateDirect(width * height * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)

        // Write pixels in RGB order
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }

        buffer.rewind()
        return buffer
    }
}
