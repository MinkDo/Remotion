package com.example.hope.mood_tracker.ui

import android.os.Build
import androidx.annotation.RequiresApi
import com.example.hope.mood_tracker.data.database.Note
import java.time.LocalDate

@RequiresApi(Build.VERSION_CODES.O)
data class NoteState constructor(
    val notes: List<Note> = emptyList(),
    val content: String = "",
    val emotion: Int = 0,
    val isAddingNote: Boolean = false,
    val date: LocalDate = LocalDate.now()
)
