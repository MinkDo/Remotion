package com.example.hope.reminder.data.database

import androidx.room.*
import kotlinx.coroutines.flow.Flow
import java.time.LocalDate

@Dao
interface TaskDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTask(task: Task): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTaskDay(taskDay: TaskDay)

    @Query("SELECT * FROM TaskDay WHERE date = :date ORDER BY isCompleted ASC, time ASC")
    suspend fun getTaskDayByDate(date: LocalDate): List<TaskDay>
    @Query("SELECT * FROM Task WHERE taskId = :taskId")
    suspend fun getTaskById(taskId: Long): Task

    @Query("SELECT * FROM TaskDay WHERE taskDayId = :taskDayId")
    suspend fun getTaskDayById(taskDayId: Long): TaskDay

    @Query("SELECT * FROM TaskDay")
    fun getAllTasks(): Flow<List<TaskDay>>
    
    @Delete
    suspend fun deleteTaskDay(taskDay: TaskDay)
    @Query("SELECT COUNT(*) FROM TaskDay WHERE taskId = :taskId")
    suspend fun countTaskDaysByTaskId(taskId: Long): Int
    @Delete
    suspend fun deleteTask(task: Task)

    @Query("UPDATE TaskDay SET isCompleted = NOT isCompleted WHERE taskDayId = :taskDayId")
    suspend fun toggleIsCompleted(taskDayId: Long)

    @Update
    suspend fun updateTaskDay(taskDay: TaskDay)
}