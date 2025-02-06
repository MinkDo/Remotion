package com.example.hope.reminder.ui

import android.annotation.SuppressLint
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.Companion.APPLICATION_KEY
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.hope.NoteApplication
import com.example.hope.reminder.data.RepeatOption
import com.example.hope.reminder.data.database.Task
import com.example.hope.reminder.data.database.TaskDay
import com.example.hope.reminder.data.repository.TaskRepository
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.time.LocalDate

class TaskViewModel(
    private val taskRepository: TaskRepository
) : ViewModel() {

    private val _tasks = taskRepository.getAllTasks()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(), emptyList())

    @RequiresApi(Build.VERSION_CODES.O)
    private val _state = MutableStateFlow(TaskState())

    @RequiresApi(Build.VERSION_CODES.O)
    val state = combine(_state, _tasks) { state, tasks ->
        state.copy(
            tasks = tasks
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), TaskState())

    @SuppressLint("NewApi")
    fun onEvent(event: TaskEvent) {
        when (event) {
            is TaskEvent.SetTitle -> {
                _state.update { it.copy(title = event.title) }
            }
            is TaskEvent.SetContent -> {
                _state.update { it.copy(content = event.content) }
            }
            is TaskEvent.SetRepeatOption -> {
                _state.update { it.copy(repeatOption = event.repeatOption) }
            }
            is TaskEvent.SetStartDate -> {
                _state.update { it.copy(startDate = event.startDate) }
            }
            is TaskEvent.SetEndDate -> {
                _state.update { it.copy(endDate = event.endDate) }
            }
            TaskEvent.ShowDialog -> {
                _state.update { it.copy(isAddingTask = true) }
            }
            TaskEvent.HideDialog -> {
                _state.update { it.copy(isAddingTask = false) }
            }
            is TaskEvent.SetTime -> {
                _state.update { it.copy(time = event.time) }
            }
            is TaskEvent.SetDateSelected -> {
                _state.update { it.copy(dateSelected = event.dateSelected) }
            }
            is TaskEvent.SetSelectedDaysOfWeek -> {
                _state.update { it.copy(selectedDaysOfWeek = event.selectedDays) }
            }

            TaskEvent.SaveTask -> {
                val title = state.value.title
                val repeatOption = state.value.repeatOption
                val startDate = state.value.startDate
                val endDate = state.value.endDate
                val time = state.value.time
                val content = state.value.content
                val selectedDays = state.value.selectedDaysOfWeek

                if (title.isBlank() || startDate == null) return

                viewModelScope.launch {
                    val taskId = taskRepository.insertTask(
                        Task(
                            title = title,
                            repeatOption = repeatOption,
                            startDate = startDate,
                            endDate = endDate
                        )
                    )
                    // Tạo TaskOccurrence dựa trên repeatOption
                    if (repeatOption == RepeatOption.DAILY) {
                        var current = startDate // Bắt đầu từ ngày startDate
                        while (!current?.isAfter(endDate)!!) { // Kiểm tra nếu current chưa vượt qua endDate
                            current.let {
                                TaskDay(
                                    taskId = taskId,
                                    date = it, // Giữ nguyên kiểu LocalDate
                                    time = time,
                                    content = content
                                )
                            }.let {
                                taskRepository.insertTaskDay(
                                    it
                                )
                            }
                            current = current.plusDays(1) // Tăng current lên 1 ngày
                        }


                    }
                    else if(repeatOption == RepeatOption.DAYINWEEK) {
                        var currentDate = startDate

                        // Tạo TaskDay cho mỗi ngày trong tuần được chọn
                        while (!currentDate?.isAfter(endDate)!!) {
                            selectedDays.forEach { dayIndex ->
                                // Điều chỉnh dayIndex từ 0 (Chủ Nhật) tới 6 (Thứ Bảy) thành 1 (Thứ Hai) tới 7 (Chủ Nhật)
                                val adjustedDayIndex = if (dayIndex == 0) 7 else dayIndex // Điều chỉnh Chủ Nhật từ 0 thành 7
                                // Tính toán ngày trong tuần (ví dụ: Thứ Hai = 1, Thứ Ba = 2,...)
                                val targetDay = currentDate?.with(
                                    java.time.temporal.WeekFields.of(java.util.Locale.getDefault())
                                        .dayOfWeek(), adjustedDayIndex.toLong()
                                )


                                if (!targetDay?.isBefore(startDate)!! && !targetDay.isAfter(endDate)) {
                                    taskRepository.insertTaskDay(
                                        TaskDay(
                                            taskId = taskId,
                                            date = targetDay,
                                            time = time,
                                            content = content
                                        )
                                    )
                                }
                            }
                            currentDate = currentDate.plusWeeks(1) // Chuyển sang tuần tiếp theo
                        }
                    }
                    else if (repeatOption == RepeatOption.MONTHLY) {
                        var current = startDate // Bắt đầu từ startDate
                        val originalDayOfMonth = startDate.dayOfMonth // Lưu lại ngày gốc (ví dụ: ngày 31 của tháng)

                        while (!current?.isAfter(endDate)!!) { // Lặp cho đến khi vượt qua endDate
                            // Lấy ngày cuối tháng
                            val lastDayOfMonth = current.withDayOfMonth(current.month.length(current.isLeapYear())) // Lấy ngày cuối tháng
                            val validDate = if (originalDayOfMonth > lastDayOfMonth.dayOfMonth) {
                                // Nếu ngày tháng hiện tại không tồn tại trong tháng tiếp theo (ví dụ ngày 31 tháng 4), lấy ngày cuối tháng
                                lastDayOfMonth
                            } else {
                                // Nếu ngày còn hợp lệ, giữ nguyên
                                current.withDayOfMonth(originalDayOfMonth)
                            }

                            // Lưu TaskDay cho ngày tính toán được
                            taskRepository.insertTaskDay(
                                TaskDay(
                                    taskId = taskId,
                                    date = validDate, // Lưu ngày đã điều chỉnh
                                    time = time,
                                    content = content
                                )
                            )

                            // Tiến đến tháng tiếp theo từ validDate
                            current = current.plusMonths(1)

                            // Kiểm tra nếu current đã vượt qua endDate, thì dừng vòng lặp
                            if (current.isAfter(endDate)) break
                        }
                    }
                    else if (repeatOption == RepeatOption.YEARLY) {
                        var current = startDate // Bắt đầu từ startDate
                        val originalDayOfMonth = startDate.dayOfMonth // Lưu lại ngày gốc (ví dụ: ngày 31 của tháng)

                        while (!current?.isAfter(endDate)!!) { // Lặp cho đến khi vượt qua endDate
                            // Lấy ngày cuối tháng
                            val lastDayOfMonth =
                                current.withDayOfMonth(current.month.length(current.isLeapYear())) // Lấy ngày cuối tháng
                            val validDate = if (originalDayOfMonth > lastDayOfMonth.dayOfMonth) {
                                // Nếu ngày tháng hiện tại không tồn tại trong tháng tiếp theo (ví dụ ngày 31 tháng 4), lấy ngày cuối tháng
                                lastDayOfMonth
                            } else {
                                // Nếu ngày còn hợp lệ, giữ nguyên
                                current.withDayOfMonth(originalDayOfMonth)
                            }

                            // Lưu TaskDay cho ngày tính toán được
                            taskRepository.insertTaskDay(
                                TaskDay(
                                    taskId = taskId,
                                    date = validDate, // Lưu ngày đã điều chỉnh
                                    time = time,
                                    content = content
                                )
                            )

                            // Tiến đến tháng tiếp theo từ validDate
                            current = current.plusYears(1)
                        }
                    }
                    else {
                        taskRepository.insertTaskDay(
                            TaskDay(
                                taskId = taskId,
                                date = startDate,
                                time = time,
                                content = content
                            )
                        )
                    }
                }

                _state.update {
                    it.copy(
                        isAddingTask = false,
                        title = "",
                        repeatOption = RepeatOption.NONE,
                        startDate = null,
                        endDate = null,
                        time = null,
                        content = "",
                        selectedDaysOfWeek = emptySet()
                    )
                }
            }

            is TaskEvent.DeleteTaskDay -> {
                viewModelScope.launch {
                    // Kiểm tra nếu Task không còn TaskDay nào liên kết
                    val taskCount = taskRepository.countTaskDaysByTaskId(event.taskDay.taskId)

                    // Nếu không còn TaskDay nào, xóa luôn Task
                    if (taskCount == 1) {
                        val task = taskRepository.getTaskById(event.taskDay.taskId)
                        taskRepository.deleteTask(task)
                    } else {
                        taskRepository.deleteTaskDay(event.taskDay)
                    }
                }
            }

            is TaskEvent.DeleteTask -> {
                viewModelScope.launch {
                    taskRepository.deleteTask(event.task)
                }
            }

            is TaskEvent.UpdateTaskDay -> {
                viewModelScope.launch {
                    taskRepository.updateTaskDay(event.taskDay)
                }
                _state.update {
                    it.copy(
                        time = null,
                        content = ""
                    )
                }
            }

            is TaskEvent.ToggleIsCompleted -> {
                viewModelScope.launch {
                    taskRepository.toggleIsCompleted(event.taskDayId)
                }
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    suspend fun getTaskByDate(date: LocalDate): List<Pair<Task, TaskDay>> {
        // Lấy tất cả TaskOccurrence cho ngày đó
        val occurrences = taskRepository.getTaskDayByDate(date)
//        val occurrences = _tasks.value.filter { it.date == date }

        // Lấy thông tin Task cho mỗi TaskOccurrence
        return occurrences.map { occurrence ->
            val task = taskRepository.getTaskById(occurrence.taskId)
            task to occurrence
        }
    }


    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val application = (this[APPLICATION_KEY] as NoteApplication)
                val taskRepository = application.container.taskRepository
                TaskViewModel(taskRepository = taskRepository)
            }
        }
    }
}

