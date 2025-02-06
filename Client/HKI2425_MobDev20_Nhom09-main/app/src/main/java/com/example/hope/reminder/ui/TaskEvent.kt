package com.example.hope.reminder.ui

import com.example.hope.reminder.data.RepeatOption
import com.example.hope.reminder.data.database.Task
import com.example.hope.reminder.data.database.TaskDay
import java.time.LocalDate
import java.time.LocalTime

sealed interface TaskEvent {
    object SaveTask : TaskEvent
    data class DeleteTaskDay(val taskDay: TaskDay): TaskEvent
    data class DeleteTask(val task: Task): TaskEvent
    data class ToggleIsCompleted(val taskDayId: Long): TaskEvent
    data class UpdateTaskDay(val taskDay: TaskDay): TaskEvent
    data class SetTitle(val title: String) : TaskEvent
    data class SetContent(val content: String?) : TaskEvent
    data class SetRepeatOption(val repeatOption: RepeatOption) : TaskEvent
    data class SetStartDate(val startDate: LocalDate?) : TaskEvent
    data class SetEndDate(val endDate: LocalDate?) : TaskEvent
    data class SetTime(val time: LocalTime?) : TaskEvent // Thời gian
    data class SetDateSelected(val dateSelected: LocalDate): TaskEvent
    data class SetSelectedDaysOfWeek(val selectedDays: Set<Int>) : TaskEvent
    object ShowDialog : TaskEvent
    object HideDialog : TaskEvent
}