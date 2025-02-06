package com.example.hope.reminder.data.repository

import android.content.Context
import android.os.Build
import androidx.annotation.RequiresApi
import com.example.hope.reminder.data.database.Task
import com.example.hope.reminder.data.database.TaskDao
import com.example.hope.reminder.data.database.TaskDay
import com.example.hope.reminder.notification.cancelTaskAlarm
import com.example.hope.reminder.notification.setTaskAlarm
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.time.LocalDate

class TaskRepositoryImpl(
    private val taskDao: TaskDao,
    private val context: Context
) : TaskRepository {

    override fun getAllTasks(): Flow<List<TaskDay>> {
        return taskDao.getAllTasks()
    }

    override suspend fun getTaskDayByDate(date: LocalDate): List<TaskDay> {
        return taskDao.getTaskDayByDate(date)
    }

    override suspend fun getTaskById(id: Long): Task {
        return taskDao.getTaskById(id)
    }

    override suspend fun insertTask(task: Task): Long {
        return taskDao.insertTask(task)
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override suspend fun insertTaskDay(taskDay: TaskDay) {
        taskDao.insertTaskDay(taskDay)
        taskDay.time?.let { _ ->
            setTaskAlarm(taskDay, context, taskDao)
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override suspend fun deleteTaskDay(taskDay: TaskDay) {
        taskDao.deleteTaskDay(taskDay)
        cancelTaskAlarm(taskDay, context)
    }

    override suspend fun deleteTask(task: Task) {
        taskDao.deleteTask(task)
    }

    override suspend fun countTaskDaysByTaskId(taskId: Long): Int {
        return taskDao.countTaskDaysByTaskId(taskId)
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override suspend fun toggleIsCompleted(taskDayId: Long) {
        val taskDay = taskDao.getTaskDayById(taskDayId)
        if (taskDay.isCompleted) {
            setTaskAlarm(taskDay,context, taskDao)
        } else {
            cancelTaskAlarm(taskDay, context)
        }
        return taskDao.toggleIsCompleted(taskDayId)
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override suspend fun updateTaskDay(taskDay: TaskDay) {
        taskDao.updateTaskDay(taskDay)
        cancelTaskAlarm(taskDay, context)
        taskDay.time?.let { _ ->
            setTaskAlarm(taskDay, context, taskDao)
        }
    }
}