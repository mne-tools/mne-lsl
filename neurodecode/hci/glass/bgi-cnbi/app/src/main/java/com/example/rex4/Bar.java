package com.example.rex4;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class Bar extends SurfaceView implements SurfaceHolder.Callback {
    private final String TAG = "Bar Class";
    private SurfaceHolder surfaceHolder;
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    final public String MESSAGE = "msg";
    BarThread thread;
    private Context ctx = null;
    public Handler mHandler;
    // f,l,u,d are direction information.
    // fill the screen with a single color. 1=red, 2=green, 3=blue, 0=no fill
    private int r, l, u, d = 0;
    private int full = 0;
    private int bar_fullcolor= Color.GREEN;

    public Bar(Context context) {
        super(context);
        surfaceHolder = getHolder();
        surfaceHolder.addCallback(this);
        ctx = context;
        setFocusable(true); // make sure we get key events

        mHandler = new Handler(Looper.getMainLooper()) {
            @Override
            public void handleMessage(Message msg) {
                this.obtainMessage();
                Bundle bundle = msg.getData();
                String s = bundle.getString(MESSAGE);
                decide(s);
            }
        };
    }

    public void decide(String s) {
        if (s != null) {
            if (s.charAt(0) == 'L') {
                l = Integer.parseInt(s.substring(1)) % (100 + 1);
            } else if (s.charAt(0) == 'R') {
                r = Integer.parseInt(s.substring(1)) % (100 + 1);
            } else if (s.charAt(0) == 'U') {
                u = Integer.parseInt(s.substring(1)) % (100 + 1);
            } else if (s.charAt(0) == 'D') {
                d = Integer.parseInt(s.substring(1)) % (100 + 1);
            } else if (s.charAt(0) == 'S') { // both sides
                l = Integer.parseInt(s.substring(1)) % (100 + 1);
                r = Integer.parseInt(s.substring(1)) % (100 + 1);
            } else if (s.charAt(0) == 'C') {
                l = 0;
                r = 0;
                u = 0;
                d = 0;
                full = 0;
            } else if (s.charAt(0) == 'F') {
                full = Integer.parseInt(s.substring(1)) % (100 + 1);
            } else if (s.charAt(0) == 'B') {
                if (s.charAt(1)=='G') bar_fullcolor= Color.GREEN;
                else if (s.charAt(1)=='R') bar_fullcolor= Color.RED;
                else if (s.charAt(1)=='B') bar_fullcolor= Color.BLUE;
                else if (s.charAt(1)=='Y') bar_fullcolor= Color.YELLOW;
            } else {
                // any other command?
            }
            //updatescr = true;
        }

    }

	public void surfaceCreated(SurfaceHolder holder) {
		Log.i(TAG, "surfaceCreated called");
		thread = new BarThread(surfaceHolder, ctx);
		thread.setRunning(true);
		thread.start();
	}

	public void surfaceChanged(SurfaceHolder holder, int format, int width,
			int height) {
		Log.i(TAG, "surfaceChanged called");
		thread.setSurfaceSize(width, height);
	}

	public void surfaceDestroyed(SurfaceHolder holder) {
		boolean retry = true;
		Log.i(TAG, "surfaceDestroyed called");
		thread.setRunning(false);
		while (retry) {
			try {
				thread.join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			retry = false;
		}
	}

	class BarThread extends Thread {
		private int canvasWidth = 200;
		private int canvasHeight = 400;
		private boolean run = false;
		private SurfaceHolder surfaceHolder;

		private float left1;
		private float top1;
		private final float height1 = 100;
		private final float width1 = 300;

		private float left2;
		private float top2;
		private final float height2 = 100;
		private final float width2 = 300;

        private float center_x;
        private float center_y;

		public BarThread(SurfaceHolder surfaceHolder, Context context) {
			this.surfaceHolder = surfaceHolder;
			ctx = context;

		}

		public void doStart() {
			synchronized (surfaceHolder) {
				// Start bubble in centre and create some random motion
				Log.i(TAG, "doStart called");
				left1 = canvasWidth / 3;
				top1 = canvasHeight / 3;
                center_x = left1 + width1 / 2;
                center_y = top1 + height1 / 2;

                left2 = left1 + (width1 / 2) - (height1 / 2);
				top2 = top1 + (width2) / 2 + height2 / 2;

			}
		}

		public void run() {
			while (run) {
				Canvas c = null;
				try {
					c = surfaceHolder.lockCanvas(null);
					synchronized (surfaceHolder) {
						// Log.i(TAG,"draw called");
						if (c != null) {
							draw(c);
						}
					}
				} finally {
					if (c != null) {
						surfaceHolder.unlockCanvasAndPost(c);
					}
				}
			}
		}

		public void setRunning(boolean b) {
			run = b;
		}

		public void setSurfaceSize(int width, int height) {
			Canvas c = null;
			try {
				c = surfaceHolder.lockCanvas(null);
				synchronized (surfaceHolder) {
					canvasWidth = width;
					canvasHeight = height;
					doStart();
				}
			} finally {
				if (c != null) {
					surfaceHolder.unlockCanvasAndPost(c);
				}
			}
		}

        /*
        private void draw(Canvas canvas) {
            // Log.i(TAG,"Inside Draw");
            // Main rect in background
            if (updatescr == false) return;
            updatescr = false;
            
            paint.setColor(Color.WHITE);
            canvas.drawRect(left1, top1, left1 + width1, top1 + height1, paint);
            canvas.drawRect(left2, top2, left2 + height2, top2 - width2, paint);
            paint.setColor(Color.BLUE);
            canvas.drawRect(left1 + (width1 / 2) - (height1 / 2), top1, left1
                    + (width1 / 2) + (height1 / 2), top1 + height1, paint);

            paint.setColor(Color.RED);
            canvas.drawRect(left1 + (width1 / 2) + (height1 / 2), top1, left1
                    + (width1 / 2) + (height1 / 2) + r, top1 + height1, paint);
            canvas.drawRect(left1 + (width1 / 2) - (height1 / 2), top1, left1
                    + (width1 / 2) - (height1 / 2) - l, top1 + height1, paint);
            canvas.drawRect(left2, top1 - u, left2 + height2, top1, paint);
            canvas.drawRect(left2, top1 + height1, left2 + height2, top1
                    + height1 + d, paint);
        }
        */

		private void draw(Canvas canvas) {
            // the following causes screen flickering. why?
            //if (updatescr == false) return;
            //Log.i(TAG, "* Updating screen *");
            //updatescr = false;

            if ( full==0 ) {
                // Main rect in background
                paint.setColor(Color.BLACK);
                canvas.drawRect(0, 0, canvasWidth, canvasHeight, paint);
                paint.setColor(Color.WHITE);
                canvas.drawRect(left1, top1, left1 + width1, top1 + height1, paint);
                canvas.drawRect(left2, top2, left2 + height2, top2 - width2, paint);
                paint.setColor(Color.BLUE);
                canvas.drawRect(left1 + (width1 / 2) - (height1 / 2), top1, left1
                        + (width1 / 2) + (height1 / 2), top1 + height1, paint);
                paint.setColor(Color.RED);
                canvas.drawCircle(center_x, center_y, 5, paint);
                //canvas.drawRect(left2+40, top1+48, left2 + 60, top1+52, paint);
                //canvas.drawRect(left2+48, top1+40, left2 + 52, top1+60, paint);

                if ( r > 0 ) {
                    if (r >= 100) paint.setColor(bar_fullcolor);
                    else paint.setColor(Color.RED);
                    canvas.drawRect(left1 + (width1 / 2) + (height1 / 2), top1, left1
                        + (width1 / 2) + (height1 / 2) + r, top1 + height1, paint);
                }

                if ( l > 0 ) {
                    if (l >= 100) paint.setColor(bar_fullcolor);
                    else paint.setColor(Color.RED);
                    canvas.drawRect(left1 + (width1 / 2) - (height1 / 2), top1, left1
                        + (width1 / 2) - (height1 / 2) - l, top1 + height1, paint);
                }

                if ( u > 0 ) {
                    if (u >= 100) paint.setColor(bar_fullcolor);
                    else paint.setColor(Color.RED);
                    canvas.drawRect(left2, top1 - u, left2 + height2, top1, paint);
                }

                if ( d > 0 ) {
                    if (d >= 100) paint.setColor(bar_fullcolor);
                    else paint.setColor(Color.RED);
                    canvas.drawRect(left2, top1 + height1, left2 + height2, top1 + height1 + d, paint);
                }
            } else {
                //paint.setStyle(Paint.Style.FILL);
                if ( full==1 ) paint.setColor(Color.RED);
                else if ( full==2 ) paint.setColor(Color.GREEN);
                else if ( full==3 ) paint.setColor(Color.BLUE);
                else if ( full==4 ) paint.setColor(Color.BLACK);
                canvas.drawRect(0, 0, canvasWidth, canvasHeight, paint);
            }
		}
	}
}
