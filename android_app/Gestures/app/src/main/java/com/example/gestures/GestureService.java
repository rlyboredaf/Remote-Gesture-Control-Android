package com.example.gestures;

//Imports
//Accessibility Service imports
import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.view.accessibility.AccessibilityEvent;

//Java imports
import java.util.Map;
import android.util.*;

//Android imports
import androidx.annotation.NonNull;

//Firebase imports
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

//Drawing imports
import android.graphics.*;
import android.view.Gravity;
import android.view.WindowManager;
import android.content.res.Resources;
import android.graphics.Point;
//------------------------------------------------------------------------------------------------------------------


public class GestureService extends AccessibilityService {
    //Class variables
    private double anchor_pos_x = 0.5, anchor_pos_y = 0.5;
    private WindowManager windowManager;
    private CursorView cursorView;
    private WindowManager.LayoutParams cursorParams;
    private final double width = Resources.getSystem().getDisplayMetrics().widthPixels;
    private final double height = Resources.getSystem().getDisplayMetrics().heightPixels;
    int v_deviation = (int) (height * 0.35);
    int h_deviation = (int) (width * 0.4);
    //------------------------------------------------------------------------------------------------------------------


    //This function takes the data snapshot from Realtime Database (RTDB) node "Pookie_anchor" and sets the anchor position.
    public void set_anchor_pos (DataSnapshot anchor_pos_snapshot) {
        @SuppressWarnings("unchecked")
        //The Node value needs to be type casted into a Map<String, Object> to access values that were set as python dictionary.
        Map <String, Object> anchor_data = (Map<String, Object>) anchor_pos_snapshot.getValue();

        if (anchor_data != null && anchor_data.containsKey("x") && anchor_data.containsKey("y")) {
            //It's safer to convert the object into a Number class object first and then taking the double value out using .doubleValue().
            Number x = (Number) anchor_data.get("x");
            Number y = (Number) anchor_data.get("y");
            if (x != null && y != null) {
                anchor_pos_x = 1 - x.doubleValue();
                anchor_pos_y = y.doubleValue();
            }
            else {
                anchor_pos_x = 0.5;
                anchor_pos_y = 0.5;
            }
            Log.i("Anchor", "x : " + anchor_pos_x + " y : " + anchor_pos_y);
        }
        else
            Log.e("Firebase", "\"anchor_data\" was null.");
    }
    //------------------------------------------------------------------------------------------------------------------


    public void perform_gesture(DataSnapshot gesture_snapshot) {
        String gesture = gesture_snapshot.getValue(String.class);
        if (gesture == null) {
            Log.e("Gesture", "The received gesture was null.");
            return;
        }
        //------------------------------------------------------------------------------------------------------------------


        //Point initialisation for path definitions.
        Point cursor = new Point((int)(anchor_pos_x * width), (int)(anchor_pos_y * height));

        //Creating the path for the gesture.
        Path path = new Path();
        path.moveTo(cursor.x, cursor.y);
        
        boolean is_gesture_recognised = true;
        switch (gesture) {
            case "swipe_up": {
                path.lineTo(cursor.x, Math.min(cursor.y - v_deviation, 0));
                Log.d("GestureService", "Received gesture: " + gesture);
                break;
            }
            case "swipe_down": {
                path.lineTo(cursor.x, Math.max(cursor.y + v_deviation, (int)height));
                Log.d("GestureService", "Received gesture: " + gesture);
                break;
            }
            case "swipe_left": {
                path.lineTo(Math.max(cursor.x - h_deviation, 0), cursor.y);
                Log.d("GestureService", "Received gesture: " + gesture);
                break;
            }
            case "swipe_right": {
                path.lineTo(Math.min(cursor.x + h_deviation, (int)width), cursor.y);
                Log.d("GestureService", "Received gesture: " + gesture);
                break;
            }
            case "tap": {
                path.lineTo(cursor.x + 1, cursor.y + 1);
                Log.d("GestureService","Received gesture: " + gesture);
                break;
            }
            default: {
                Log.d("GestureService", "Received unknown gesture: " + gesture);
                is_gesture_recognised = false;
            }
        }
        //------------------------------------------------------------------------------------------------------------------


        if (is_gesture_recognised) {
            //If it is recognised dispatch the path as a gesture using the dispatchGesture() method.

            //Build the gesture
            GestureDescription.StrokeDescription swipe = new GestureDescription.StrokeDescription(path, 0, 300);
            GestureDescription.Builder builder = new GestureDescription.Builder();
            builder.addStroke(swipe);
            GestureDescription performed_gesture = builder.build();
            //------------------------------------------------------------------------------------------------------------------


            boolean is_dispatch_successful = dispatchGesture(performed_gesture, null, null);
            if (!is_dispatch_successful)
                Log.e("Gesture_dispatch", "class: perform_gesture, failed to dispatch gesture: " + gesture);
        }
    }
    //------------------------------------------------------------------------------------------------------------------


    //Some method that I don't recognise. I only know that this is the one that's called when the Accessibility Service is started.
    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
    }
    //------------------------------------------------------------------------------------------------------------------


    //Another method that I don't recognise. Apparently, this one activates when the service (app) is closed?
    @Override
    public void onInterrupt() {
        // --- Clean up the cursor when the service is stopped ---
        if (windowManager != null && cursorView != null) {
            windowManager.removeView(cursorView);
        }
    }
    //------------------------------------------------------------------------------------------------------------------


    //When you finally approve the accessibility permission this is the method that gets called.
    @Override
    public void onServiceConnected() {
        super.onServiceConnected();
        //Getting firebase references to the nodes in database. Or inshort establishing connection to the nodes.
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference gesture = database.getReference("Pookie");
        DatabaseReference anchor_pos = database.getReference("Pookie_anchor");
        //------------------------------------------------------------------------------------------------------------------


        //Window thingies. These manage the Cursor displaying.
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        cursorView = new CursorView(this);
        cursorParams = new WindowManager.LayoutParams(75, 75, WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY, WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, PixelFormat.TRANSLUCENT);
        cursorParams.gravity = Gravity.TOP | Gravity.START;
        //Adds the view we made up above.
        windowManager.addView(cursorView, cursorParams);
        //------------------------------------------------------------------------------------------------------------------


        //The data change listener to the "Pookie_anchor" node.
        anchor_pos.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                //Calls the method that'll update the anchor position.
                set_anchor_pos(snapshot);
                //------------------------------------------------------------------------------------------------------------------


                //Updates the cursor's position according to the anchor coords received.
                if (cursorView != null) {
                    // Calculate the new pixel position
                    cursorParams.x = (int) (anchor_pos_x * width) - (cursorParams.width / 2);
                    cursorParams.y = (int) (anchor_pos_y * height) - (cursorParams.height / 2);

                    // Tell the WindowManager to apply the new position
                    windowManager.updateViewLayout(cursorView, cursorParams);
                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                Log.e("Anchor", "The value event listener was cancelled.");
                Log.e("Anchor", "Database error: " + error.getMessage());
                Log.e("Anchor", "Error Code: " + error.getCode());
                Log.e("Anchor", "Error Details: " + error.getDetails());
            }
        });
        //------------------------------------------------------------------------------------------------------------------


        //The data change listener to the "Pookie" node.
        gesture.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                String received_gesture = snapshot.getValue(String.class);
                if (received_gesture != null)
                    perform_gesture(snapshot);
                else
                    Log.e("Gesture", "received_gesture was null.");
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                Log.e("Gesture", "The value event listener for gesture was cancelled.");
                Log.e("Gesture", "Database error: " + error.getMessage());
                Log.e("Gesture", "Error Code: " + error.getCode());
                Log.e("Gesture", "Error Details: " + error.getDetails());
            }
        });
    }
}
