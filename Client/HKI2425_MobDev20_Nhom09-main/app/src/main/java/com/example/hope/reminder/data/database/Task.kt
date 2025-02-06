package com.example.hope.reminder.data.database

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.PrimaryKey
import androidx.room.ForeignKey.Companion.CASCADE
import com.example.hope.reminder.data.RepeatOption
import java.time.LocalDate
import java.time.LocalTime

@Entity
data class Task(
    @PrimaryKey(autoGenerate = true)
    val taskId: Long = 0,
    val title: String,
    val repeatOption: RepeatOption, // "NONE", "DAILY"
    val startDate: LocalDate?,     // Ngày bắt đầu (timestamp, null nếu không lặp lại)
    val endDate: LocalDate?        // Ngày kết thúc (timestamp, null nếu không lặp lại)
)

@Entity(
    foreignKeys = [ForeignKey(
        entity = Task::class,
        parentColumns = ["taskId"],
        childColumns = ["taskId"],
        onDelete = CASCADE // Khi xóa Task, tự động xóa các TaskOccurrence liên quan
    )]
)
data class TaskDay(
    @PrimaryKey(autoGenerate = true)
    val taskDayId: Long = 0,
    val taskId: Long, // Liên kết tới Task
    val date: LocalDate,   // Ngày xuất hiện (timestamp)
    val time: LocalTime?, // Giờ thực hiện (có thể null nếu không cần giờ cụ thể)
    val content: String?, // Nội dung chi tiết (có thể khác từng ngày)
    val isCompleted: Boolean = false // Trạng thái hoàn thành cho từng ngày
)
