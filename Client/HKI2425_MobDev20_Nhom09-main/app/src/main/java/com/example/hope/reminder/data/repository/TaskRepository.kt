package com.example.hope.reminder.data.repository

import com.example.hope.reminder.data.database.Task
import com.example.hope.reminder.data.database.TaskDay
import kotlinx.coroutines.flow.Flow
import java.time.LocalDate

interface TaskRepository {
    fun getAllTasks(): Flow<List<TaskDay>>
    suspend fun getTaskDayByDate(date: LocalDate): List<TaskDay>
    suspend fun getTaskById(id: Long): Task
    suspend fun insertTask(task: Task): Long
    suspend fun insertTaskDay(taskDay: TaskDay)
    suspend fun deleteTaskDay(taskDay: TaskDay)
    suspend fun countTaskDaysByTaskId(taskId: Long): Int
    suspend fun deleteTask(task: Task)
    suspend fun toggleIsCompleted(taskDayId: Long)
    suspend fun updateTaskDay(taskDay: TaskDay)
}