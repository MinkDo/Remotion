package com.example.hope.reminder.notification

import android.annotation.SuppressLint
import android.app.AlarmManager
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class SnoozeNotificationReceiver : BroadcastReceiver() {
    @SuppressLint("ScheduleExactAlarm", "ServiceCast")
    override fun onReceive(context: Context, intent: Intent) {

        val taskDayId = intent.getLongExtra("taskDayId", -1)
        val title = intent.getStringExtra("title") ?: "No Title"
        val content = intent.getStringExtra("content") ?: "No Content"

        // Lấy giá trị thời gian snooze từ SharedPreferences
        val sharedPreferences = context.getSharedPreferences("app_preferences", Context.MODE_PRIVATE)
        val snoozeTimeGet = sharedPreferences.getInt("snooze_time", 10) // Mặc định 10 phút nếu chưa cài đặt

        // Cập nhật lại báo thức sau 10 phút
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val taskAlarmIntent = Intent(context, TaskAlarmReceiver::class.java).apply {
            putExtra("taskDayId", taskDayId)
            putExtra("title", title) // Ví dụ title
            putExtra("content", content)
        }

        val pendingIntent = PendingIntent.getBroadcast(
            context,
            taskDayId.toInt(),
            taskAlarmIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val snoozeTime = System.currentTimeMillis() + snoozeTimeGet * 60 * 1000 // Chuyển đổi phút thành millis

        alarmManager.setExactAndAllowWhileIdle(
            AlarmManager.RTC_WAKEUP,
            snoozeTime,
            pendingIntent
        )

        // Xóa thông báo hiện tại
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.cancel(taskDayId.toInt())  // Xóa thông báo cũ khi báo lại
    }
}
