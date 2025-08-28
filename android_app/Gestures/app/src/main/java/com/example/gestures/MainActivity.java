package com.example.gestures;

//Android imports
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;
import android.text.TextUtils;
import android.util.*;
import android.view.View;
import android.widget.*;

//Some other totally not dubious android imports
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

public class MainActivity extends AppCompatActivity {
	private Button teleporter_button;
	private TextView status;

	@SuppressLint("MissingInflatedId")
    @Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		status = findViewById(R.id.statusText);
		teleporter_button = findViewById(R.id.teleporter);

		teleporter_button.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				Intent intent = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
				startActivity(intent);
			}
		});
	}
	@Override
	protected void onResume() {
		super.onResume();
		updateStatus();
	}

	protected void updateStatus() {
		if (isAccessibilityServiceEnabled(this, GestureService.class)) {
			status.setText("CONNECTED");
			status.setTextColor(ContextCompat.getColor(this, android.R.color.holo_green_light));
			Log.d("MainActivity", "Gesture service is enabled.");
		} else {
			status.setText("DISCONNECTED");
			status.setTextColor(ContextCompat.getColor(this, android.R.color.holo_red_light));
			Log.d("MainActivity", "Gesture service is disabled.");
		}
	}
	public static boolean isAccessibilityServiceEnabled(Context context, Class<?> accessibilityService) {
		TextUtils.SimpleStringSplitter colonSplitter = new TextUtils.SimpleStringSplitter(':');
		String settingValue = Settings.Secure.getString(
				context.getApplicationContext().getContentResolver(),
				Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES);

		if (settingValue != null) {
			colonSplitter.setString(settingValue);
			while (colonSplitter.hasNext()) {
				String componentName = colonSplitter.next();
				if (componentName.equalsIgnoreCase(context.getPackageName() + "/" + accessibilityService.getName())) {
					return true;
				}
			}
		}
		return false;
	}
};