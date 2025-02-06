package com.example.hope.reminder.notification

import android.annotation.SuppressLint
import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.annotation.RequiresApi
import com.example.hope.reminder.data.database.TaskDao
import com.example.hope.reminder.data.database.TaskDay
import java.time.Duration
import java.time.LocalDateTime
import java.time.LocalTime

@RequiresApi(Build.VERSION_CODES.O)
@SuppressLint("ServiceCast", "ScheduleExactAlarm")
suspend fun setTaskAlarm(taskDay: TaskDay, context: Context, taskDao: TaskDao) {

    val task = taskDao.getTaskById(taskDay.taskId)

    val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
    val intent = Intent(context, TaskAlarmReceiver::class.java).apply {
        putExtra("taskDayId", taskDay.taskDayId)
        putExtra("title", "Task: ${task.title}") // Ví dụ title
        putExtra("content", taskDay.content ?: "No content")
    }

    val pendingIntent = PendingIntent.getBroadcast(
        context,
        taskDay.taskId.toInt(),
        intent,
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
    )

    // Tính toán thời gian báo thức
    val targetTime = LocalDateTime.of(taskDay.date, taskDay.time?.withSecond(0)?.withNano(0) ?: LocalTime.NOON.withSecond(0).withNano(0))
    val delayMillis = Duration.between(LocalDateTime.now(), targetTime).toMillis()

    // Đặt alarm nếu thời gian hợp lệ
    if (delayMillis > 0) {
        alarmManager.setExactAndAllowWhileIdle(
            AlarmManager.RTC_WAKEUP,
            System.currentTimeMillis() + delayMillis,
            pendingIntent
        )
    }
}

@RequiresApi(Build.VERSION_CODES.O)
fun cancelTaskAlarm(taskDay: TaskDay, context: Context) {
    // Hủy thông báo
    val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
    val intent = Intent(context, TaskAlarmReceiver::class.java).apply {
        putExtra("taskId", taskDay.taskId)
    }
    val pendingIntent = PendingIntent.getBroadcast(
        context,
        taskDay.taskId.toInt(),
        intent,
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
    )

    alarmManager.cancel(pendingIntent)
}