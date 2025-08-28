// In a new file: CursorView.java
package com.example.gestures;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

import androidx.annotation.NonNull;

public class CursorView extends View {
    private final Paint cursorPaint;

    public CursorView(Context context) {
        super(context);
        // Set up the Paint object for our circle
        cursorPaint = new Paint();
        cursorPaint.setColor(Color.parseColor("#7A5CFF")); // You can choose any color
        cursorPaint.setAlpha(128); // Make it semi-transparent (0-255)
        cursorPaint.setAntiAlias(true);
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);
        // Draw a circle in the center of this View's canvas
        float centerX = getWidth() / 2f;
        float centerY = getHeight() / 2f;
        float radius = getWidth() / 2f;
        canvas.drawCircle(centerX, centerY, radius, cursorPaint);
    }
}