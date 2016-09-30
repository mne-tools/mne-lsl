package com.example.rex4;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

import android.app.Activity;
import android.os.Bundle;
import android.os.Message;
import android.util.Log;
import android.view.WindowManager;
import android.widget.TextView;

public class MainActivity extends Activity {

	final public String TAG = "MainActivity";
	final public String MESSAGE = "msg";
	private TextView mText;
	private MyServer server;
	private String message;
	private Message msg;
	private boolean serverOn;
	private Bar bar;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
		bar = new Bar(this);
		setContentView(bar);
		msg = new Message();

		serverOn = true;
		server = new MyServer();
		server.start();

	}

	public void setText(String message) {
		mText.setText(message);
	}

	private class MyServer extends Thread {

		private ServerSocket mSocket;
		private Socket client;
		private BufferedReader bufferedReader;
		private MyRunnable myRun;

		@Override
		public void run() {
			try {
				mSocket = new ServerSocket(59900);
				myRun = new MyRunnable();
			} catch (IOException e) {
				Log.w(TAG, "Server faield to start,socket error");
				e.printStackTrace();
			}

			Log.i(TAG, "Server started ...");

			while (serverOn) {
				try {
					client = mSocket.accept();
					bufferedReader = new BufferedReader(new InputStreamReader(
							client.getInputStream()));

					while ((message = bufferedReader.readLine()) != null) {

						Log.i("MainActiv", "Message received from client: "
								+ message);
//						Bundle bundle = new Bundle();
//						bundle.putString(MESSAGE, message);
//						msg.setData(bundle);
//						bar.mHandler.sendMessage(msg);
						bar.mHandler.post(myRun);
					}
					bufferedReader.close();
					client.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}

	public class MyRunnable implements Runnable{

		@Override
		public void run() {
			bar.decide(message);
			
		}
		
	}
	@Override
	protected void onPause() {
		serverOn = false;
		if (server != null) {
			server.interrupt();
		}
		super.onPause();
	}
}
