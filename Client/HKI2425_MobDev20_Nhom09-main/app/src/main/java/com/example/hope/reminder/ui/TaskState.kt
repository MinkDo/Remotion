package com.example.hope.reminder.ui

import android.os.Build
import androidx.annotation.RequiresApi
import com.example.hope.reminder.data.RepeatOption
import com.example.hope.reminder.data.database.Task
import com.example.hope.reminder.data.database.TaskDay
import java.time.LocalDate
import java.time.LocalTime

@RequiresApi(Build.VERSION_CODES.O)
data class TaskState (
    val tasks: List<TaskDay> = emptyList(),
    val title: String = "",
    var content: String? = "",
    val repeatOption: RepeatOption = RepeatOption.NONE,
    val startDate: LocalDate? = null,
    val endDate: LocalDate? = null,
    var time: LocalTime? = null,
    val isAddingTask: Boolean = false,
    val dateSelected: LocalDate = LocalDate.now(),
    val selectedDaysOfWeek: Set<Int> = emptySet() // Thêm mảng lưu các ngày đã chọn
)
