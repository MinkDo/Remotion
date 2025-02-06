package com.example.hope.mood_tracker.data.database

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.room.Entity
import androidx.room.PrimaryKey
import java.time.LocalDate

@Entity
@RequiresApi(Build.VERSION_CODES.O)
data class Note constructor(
    val userId: String = "",
    @PrimaryKey
    val date: LocalDate = LocalDate.now(),
    val content: String = "",
    val emotion: Int = 0
)
