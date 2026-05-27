package com.coralreefdetector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImageUtils {

    /** Load image from path */
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

    /** Center crop (important to match training) */
    private fun centerCrop(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        val newSize = minOf(width, height)

        val xOffset = (width - newSize) / 2
        val yOffset = (height - newSize) / 2

        return Bitmap.createBitmap(bitmap, xOffset, yOffset, newSize, newSize)
    }

    /** Convert bitmap to Float buffer for model */
    fun bitmapToFloatBuffer(bitmap: Bitmap, width: Int = 224, height: Int = 224): ByteBuffer {

        // STEP 1: center crop
        val cropped = centerCrop(bitmap)

        // STEP 2: resize
        val resized = Bitmap.createScaledBitmap(cropped, width, height, true)

        // STEP 3: buffer
        val buffer = ByteBuffer.allocateDirect(width * height * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)

        // STEP 4: normalize [0,1]
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
