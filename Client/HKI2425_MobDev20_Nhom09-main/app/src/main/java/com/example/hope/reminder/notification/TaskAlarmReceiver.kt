package com.example.hope.reminder.notification

import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import com.example.hope.MainActivity
import com.example.hope.R

class TaskAlarmReceiver : BroadcastReceiver() {
    @SuppressLint("ServiceCast", "ScheduleExactAlarm")
    override fun onReceive(context: Context, intent: Intent) {
        val sharedPreferences = context.getSharedPreferences("app_preferences", Context.MODE_PRIVATE)
        val showAlarmScreen = sharedPreferences.getBoolean("is_notification_enabled", false)

        val taskDayId = intent.getLongExtra("taskDayId", -1)
        val title = intent.getStringExtra("title") ?: "No Title"
        val content = intent.getStringExtra("content") ?: "No Content"

        if (showAlarmScreen) {
            showNotification(context, taskDayId, title, content)
        }
    }

    @SuppressLint("ServiceCast")
    private fun showNotification(context: Context, taskDayId: Long, title: String, content: String) {
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channelId = "task_notification_channel"

        // Tạo channel nếu chưa có (Android 8.0+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "Task Notifications",
                NotificationManager.IMPORTANCE_HIGH
            )
            notificationManager.createNotificationChannel(channel)
        }

        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }

        val pendingIntent = PendingIntent.getActivity(
            context,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        // Nút "Dismiss"
        val dismissIntent = getDismissPendingIntent(context, taskDayId)
        // Nút "Snooze" (Báo lại sau 10 phút)
        val snoozeIntent = getSnoozePendingIntent(context, taskDayId, title, content)

        val notification = NotificationCompat.Builder(context, channelId)
            .setSmallIcon(R.drawable.app_logo_big_eb82ef2b)  // Sử dụng icon phù hợp
            .setContentTitle(title)
            .setContentText(content)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .addAction(R.drawable.ic_launcher_foreground, "Tắt", dismissIntent)
            .addAction(R.drawable.ic_launcher_foreground, "Báo lại", snoozeIntent)
            .build()

        notificationManager.notify(taskDayId.toInt(), notification)
    }

    // PendingIntent để "Dismiss"
    private fun getDismissPendingIntent(context: Context, taskDayId: Long): PendingIntent {
        val intent = Intent(context, DismissNotificationReceiver::class.java).apply {
            putExtra("taskDayId", taskDayId)
        }
        return PendingIntent.getBroadcast(
            context,
            taskDayId.toInt(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    // PendingIntent để báo lại sau 10 phút
    private fun getSnoozePendingIntent(context: Context, taskDayId: Long, title: String, content: String): PendingIntent {
        val intent = Intent(context, SnoozeNotificationReceiver::class.java).apply {
            putExtra("taskDayId", taskDayId)
            putExtra("title", title) // Ví dụ title
            putExtra("content", content)
        }
        return PendingIntent.getBroadcast(
            context,
            taskDayId.toInt(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }
}

